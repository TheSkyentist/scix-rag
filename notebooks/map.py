import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Map of the corpus

    Every paper is one point; papers that read alike sit close together, so topics form
    **islands**. Position comes from the mean of a paper's chunk embeddings, projected to 2D with
    UMAP (cached in `data/paper_map.npz`, rebuilt when `embed` has added vectors).
    Scroll to zoom, drag to pan, hover for the paper. Use the lasso/box tool (toolbar) to
    select papers and list them below.
    """)
    return


@app.cell
def _():
    import sys

    import duckdb
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    from scipy.ndimage import gaussian_filter

    return duckdb, gaussian_filter, go, mo, np, pl, sys


@app.cell
def _(duckdb, mo, sys):
    root = mo.notebook_dir().parent
    sys.path.insert(0, str(root))  # for `config` and `src.paper_map`
    db_path = root / 'data' / 'papers.duckdb'
    mo.stop(
        not db_path.exists(),
        mo.md(f'No database at `{db_path}`. Run `pixi run query` first.'),
    )
    try:
        con = duckdb.connect(str(db_path), read_only=True)
    except duckdb.IOException as e:
        mo.stop(
            True,
            mo.md(
                f'Cannot open the database (is `embed` or `chunk` running?)\n\n`{e}`'
            ),
        )
    return (con,)


@app.cell
def _(con, mo, pl):
    from src import paper_map

    with mo.status.spinner('Building the map (first run only: ~1 min)...'):
        pm = paper_map.load(con)
        meta = (
            pl.DataFrame({'bibcode': pm.bibcodes})
            .join(
                con.sql(
                    'SELECT bibcode, title, first_author, year, citation_count, '
                    'arxiv_class, keyword FROM papers'
                ).pl(),
                on='bibcode',
                how='left',
            )
            .with_columns(
                pl.col('year').fill_null(pl.col('year').max()),
                pl.col('citation_count').fill_null(0),
                # arXiv lists the primary category first
                pl.col('arxiv_class')
                .list.first()
                .fill_null('(no arXiv class)')
                .alias('arxiv_primary'),
            )
        )
        island_id = paper_map.find_islands(pm.xy)
        island_names = paper_map.name_islands(
            island_id, meta['title'].to_list(), meta['keyword'].to_list()
        )
        index = pl.DataFrame({'bibcode': pm.bibcodes, 'i': range(len(pm.bibcodes))})
        edges = (
            con.sql('SELECT citing_bibcode, cited_bibcode FROM citation_edges')
            .pl()
            .join(
                index.rename({'bibcode': 'citing_bibcode', 'i': 'src'}),
                on='citing_bibcode',
            )
            .join(
                index.rename({'bibcode': 'cited_bibcode', 'i': 'dst'}),
                on='cited_bibcode',
            )
        )
    return edges, island_id, island_names, meta, pm


@app.cell
def _(meta, mo):
    _years = (int(meta['year'].min()), int(meta['year'].max()))
    color_by = mo.ui.dropdown(
        ['year', 'citations', 'island', 'arXiv class'],
        value='year',
        label='Colour',
    )
    year_max = mo.ui.slider(
        _years[0], _years[1], value=_years[1], step=1, label='Papers up to', show_value=True
    )
    edge_mode = mo.ui.dropdown(
        ['off', 'island flows', 'all papers'], value='off', label='Citations'
    )
    size_by_cites = mo.ui.checkbox(label='Size by citations')
    show_density = mo.ui.checkbox(value=True, label='Density')
    n_labels = mo.ui.slider(0, 30, value=20, label='Island labels', show_value=True)
    search = mo.ui.text(placeholder='title words / author / bibcode', label='Focus paper')
    return (
        color_by,
        edge_mode,
        n_labels,
        search,
        show_density,
        size_by_cites,
        year_max,
    )


@app.cell
def _(meta, mo, pl, search):
    _q = search.value.strip().lower()
    if _q:
        _hits = meta.filter(
            pl.col('title').str.to_lowercase().str.contains(_q, literal=True)
            | pl.col('first_author').str.to_lowercase().str.contains(_q, literal=True)
            | pl.col('bibcode').str.to_lowercase().str.contains(_q, literal=True)
        ).sort('citation_count', descending=True, nulls_last=True)
        _options = {
            f'{r["first_author"]} {r["year"]}: {(r["title"] or "")[:70]}': r['bibcode']
            for r in _hits.head(25).iter_rows(named=True)
        }
    else:
        _options = {}
    focus = mo.ui.dropdown(_options, label=f'{len(_options)} match(es)')
    return (focus,)


@app.cell
def _(
    color_by,
    edge_mode,
    edges,
    focus,
    gaussian_filter,
    go,
    island_id,
    island_names,
    meta,
    mo,
    n_labels,
    np,
    pm,
    show_density,
    size_by_cites,
    year_max,
):
    # Everything is built inside a function so the dozens of temporaries stay local
    # (marimo would otherwise treat each as a notebook-wide variable).
    def _build():
        MAX_LINES = 15_000
        dark = mo.app_meta().theme == 'dark'
        ink = '#c3c2b7' if dark else '#52514e'
        muted = '#9a9993'
        # categorical slots from the project palette (light / dark steps), then "other"
        cat = (
            ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']
            if dark
            else ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
        )  # fmt: skip
        # sequential blue (100->700); the light end recedes on the surface, so each mode
        # drops the end that would vanish
        ramp = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
                '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']  # fmt: skip
        seq = ramp[:10][::-1] if dark else ramp[3:]

        x, y = pm.xy[:, 0], pm.xy[:, 1]
        visible = np.flatnonzero(meta['year'].to_numpy() <= year_max.value)
        focus_i = pm.bibcodes.index(focus.value) if focus.value else None

        # ---- colours
        def sequential(values, label):
            all_v = np.asarray(values, dtype=float)
            lo, hi = np.percentile(all_v, [2, 98])  # a few outliers must not flatten the ramp
            return dict(
                color=all_v[visible],
                colorscale=[[k / (len(seq) - 1), c] for k, c in enumerate(seq)],
                cmin=lo,
                cmax=hi,
                colorbar=dict(title=dict(text=label, font=dict(color=ink)),
                              tickfont=dict(color=ink), thickness=10, len=0.5),
            )  # fmt: skip

        legend_items = []  # (label, colour): drawn as empty traces so the legend is real
        if color_by.value == 'year':
            marker_color = sequential(meta['year'].to_numpy(), 'year')
        elif color_by.value == 'citations':
            marker_color = sequential(np.log10(1 + meta['citation_count'].to_numpy()), 'log10(1 + citations)')
        else:
            if color_by.value == 'island':
                keys = np.array([island_names.get(c, '') for c in island_id])
                order = sorted(island_names, key=lambda c: -(island_id == c).sum())
                top = [island_names[c] for c in order[:8]]
            else:
                keys = meta['arxiv_primary'].to_numpy()
                vals, counts = np.unique(keys, return_counts=True)
                top = [str(v) for v in vals[np.argsort(-counts)][:7]]
            lut = {k: cat[i] for i, k in enumerate(top)}
            legend_items = [(k, lut[k]) for k in top] + [('other', muted)]
            marker_color = dict(color=[lut.get(k, muted) for k in keys[visible]])

        # ---- sizes
        if size_by_cites.value:
            size = 3 + 2.2 * np.log10(1 + meta['citation_count'].to_numpy()[visible])
        else:
            size = 4

        # ---- hover text
        titles = meta['title'].to_list()
        authors = meta['first_author'].to_list()
        pub_years = meta['year'].to_list()
        n_cites = meta['citation_count'].to_list()

        def hover(i):
            t = titles[i] or ''
            t = t if len(t) <= 64 else t[:61] + '...'
            isl = island_names.get(int(island_id[i]), '').split(' · ')[0]
            return f'<b>{t}</b><br>{authors[i]} {pub_years[i]} · {n_cites[i]} cit. · {isl}'

        text = [hover(i) for i in visible]

        fig = go.Figure()

        # ---- density "sea level": land where papers pile up
        if show_density.value and len(visible) > 10:
            pad = 1.0
            lo, hi = pm.xy.min(0) - pad, pm.xy.max(0) + pad
            grid, xe, ye = np.histogram2d(
                x[visible], y[visible], bins=90, range=[[lo[0], hi[0]], [lo[1], hi[1]]]
            )
            z = gaussian_filter(grid.T, sigma=1.6)
            land = '255,255,255' if dark else '60,60,50'
            fig.add_trace(go.Contour(
                x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, z=z,
                ncontours=9, contours=dict(coloring='fill', showlines=False, start=z.max() * 0.04),
                colorscale=[[0, f'rgba({land},0)'], [1, f'rgba({land},0.30)']],
                showscale=False, hoverinfo='skip', line_width=0,
            ))  # fmt: skip

        # ---- citation edges
        n_edges = 0
        edge_note = ''
        if edge_mode.value != 'off':
            src, dst = edges['src'].to_numpy(), edges['dst'].to_numpy()
            shown = np.zeros(len(pm.bibcodes), dtype=bool)
            shown[visible] = True
            keep = shown[src] & shown[dst]
            src, dst = src[keep], dst[keep]
            n_shown_edges = len(src)
            sa, sb = island_id[src], island_id[dst]
            centre = {
                c: (float(np.median(x[island_id == c])), float(np.median(y[island_id == c])))
                for c in island_names
            }

            if edge_mode.value == 'island flows':
                # How many citations run between each pair of islands (either direction):
                # one line per pair, width ~ count. Within-island and open-water edges are left out.
                both = (sa >= 0) & (sb >= 0) & (sa != sb)
                pair_counts: dict[tuple[int, int], int] = {}
                for a, b in zip(sa[both].tolist(), sb[both].tolist()):
                    k = (min(a, b), max(a, b))
                    pair_counts[k] = pair_counts.get(k, 0) + 1
                top_pairs = sorted(pair_counts.items(), key=lambda kv: -kv[1])[:60]
                n_edges = sum(n for _, n in top_pairs)
                edge_note = f'{n_edges:,} citations between islands (top {len(top_pairs)} pairs of islands)'
                busiest = top_pairs[0][1] if top_pairs else 1
                mid_x, mid_y, mid_t = [], [], []
                for (a, b), n in top_pairs:
                    (ax, ay), (bx, by) = centre[a], centre[b]
                    fig.add_trace(go.Scatter(
                        x=[ax, bx], y=[ay, by], mode='lines', hoverinfo='skip', showlegend=False,
                        line=dict(width=0.6 + 7 * n / busiest, color='rgba(235,104,52,0.45)'),
                    ))  # fmt: skip
                    mid_x.append((ax + bx) / 2)
                    mid_y.append((ay + by) / 2)
                    mid_t.append(f'<b>{n:,} citations</b><br>{island_names[a]}<br>↔ {island_names[b]}')
                fig.add_trace(go.Scatter(
                    x=mid_x, y=mid_y, mode='markers', text=mid_t, hoverinfo='text', showlegend=False,
                    marker=dict(size=7, color='rgba(235,104,52,0.9)'),
                ))  # fmt: skip
            else:
                def segments(a, b):
                    sx, sy = np.full(3 * len(a), np.nan), np.full(3 * len(a), np.nan)
                    sx[0::3], sx[1::3] = x[a], x[b]
                    sy[0::3], sy[1::3] = y[a], y[b]
                    return sx, sy

                # open water (-1) belongs to no island, so its edges count as neither
                cross = (sa != sb) & (sa >= 0) & (sb >= 0)
                # Every line is sent to the browser as data: past ~15K the page gets heavy and
                # the picture is already solid, so draw a fixed random sample.
                if len(src) > MAX_LINES:
                    pick = np.random.default_rng(0).choice(len(src), MAX_LINES, replace=False)
                    src, dst, cross = src[pick], dst[pick], cross[pick]
                plain = f'rgba({"255,255,255" if dark else "60,60,50"},0.05)'
                for mask, colour in [(~cross, plain), (cross, 'rgba(235,104,52,0.12)')]:
                    sx, sy = segments(src[mask], dst[mask])
                    n_edges += int(mask.sum())
                    fig.add_trace(go.Scattergl(
                        x=sx, y=sy, mode='lines', line=dict(width=0.5, color=colour),
                        hoverinfo='skip', showlegend=False,
                    ))  # fmt: skip
                edge_note = f'{n_edges:,} of {n_shown_edges:,} citation lines (orange = between islands)'
                if n_edges < n_shown_edges:
                    edge_note += ', random sample'

        # ---- focus paper: its references and citations, in context
        if focus_i is not None:
            fb = pm.bibcodes[focus_i]
            refs = edges.filter(edges['citing_bibcode'] == fb)['dst'].to_numpy()
            cites = edges.filter(edges['cited_bibcode'] == fb)['src'].to_numpy()
            for ids, name, colour, outgoing in [
                (refs, 'references', '#eb6834', True),
                (cites, 'cited by', '#1baf7a', False),
            ]:
                if not len(ids):
                    continue
                sx, sy = np.full(3 * len(ids), np.nan), np.full(3 * len(ids), np.nan)
                sx[0::3], sx[1::3] = x[focus_i], x[ids]
                sy[0::3], sy[1::3] = y[focus_i], y[ids]
                fig.add_trace(go.Scattergl(x=sx, y=sy, mode='lines', hoverinfo='skip',
                                           line=dict(width=1.2, color=colour), showlegend=False))  # fmt: skip
                fig.add_trace(go.Scattergl(
                    x=x[ids], y=y[ids], mode='markers', name=name,
                    marker=dict(size=8, color=colour, line=dict(width=1, color='#fff' if dark else '#1a1a19')),
                    text=[hover(i) for i in ids], hoverinfo='text',
                ))  # fmt: skip

        # ---- the papers (the trace `selected` below reads by curve number)
        node_curve = len(fig.data)
        fig.add_trace(go.Scattergl(
            x=x[visible], y=y[visible], mode='markers', name='papers', showlegend=False,
            marker=dict(size=size, opacity=0.8, line=dict(width=0), **marker_color),
            text=text, hoverinfo='text',
            hoverlabel=dict(align='left'),
        ))  # fmt: skip
        for name, colour in legend_items:
            fig.add_trace(go.Scattergl(x=[None], y=[None], mode='markers', name=name,
                                       marker=dict(size=8, color=colour)))  # fmt: skip
        if focus_i is not None:
            fig.add_trace(go.Scattergl(
                x=[x[focus_i]], y=[y[focus_i]], mode='markers', hoverinfo='text',
                text=[hover(focus_i)], showlegend=False,
                marker=dict(size=14, color='rgba(0,0,0,0)',
                            line=dict(width=2.5, color='#fff' if dark else '#1a1a19')),
            ))  # fmt: skip

        # ---- island names at their medians, biggest islands first
        ids_by_size = sorted(island_names, key=lambda c: -(island_id == c).sum())[: n_labels.value]
        lab = [(np.median(x[island_id == c]), np.median(y[island_id == c]), island_names[c]) for c in ids_by_size]
        surface = 'rgba(26,26,25,0.72)' if dark else 'rgba(252,252,251,0.72)'
        fig.update_layout(annotations=[
            dict(x=a, y=b, text=' · '.join(t.split(' · ')[:2]), showarrow=False,
                 font=dict(size=11, color=ink), bgcolor=surface, borderpad=2)
            for a, b, t in lab
        ])  # fmt: skip

        fig.update_layout(
            height=780, margin=dict(l=0, r=0, t=0, b=0), dragmode='pan', uirevision='map',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'),
            legend=dict(font=dict(color=ink), bgcolor='rgba(0,0,0,0)'),
        )  # fmt: skip
        plot = mo.ui.plotly(fig, config={'scrollZoom': True, 'displaylogo': False})
        return edge_note, node_curve, plot, visible

    edge_note, node_curve, plot, visible = _build()
    return edge_note, node_curve, plot, visible


@app.cell
def _(
    color_by,
    edge_mode,
    edge_note,
    focus,
    mo,
    n_labels,
    plot,
    search,
    show_density,
    size_by_cites,
    visible,
    year_max,
):
    _note = f'{len(visible):,} papers' + (f', {edge_note}' if edge_note else '')
    mo.vstack(
        [
            mo.hstack([color_by, year_max, edge_mode], justify='start', gap=2),
            mo.hstack([size_by_cites, show_density, n_labels, search, focus],
                      justify='start', gap=2, align='center'),
            mo.md(f'*{_note}*'),
            plot,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selected papers
    """)
    return


@app.cell
def _(island_id, island_names, meta, mo, node_curve, pl, plot, visible):
    _picked = sorted(
        {
            int(visible[p['pointIndex']])
            for p in plot.value
            if p.get('curveNumber') == node_curve and p.get('pointIndex') is not None
        }
    )
    mo.stop(
        not _picked,
        mo.md('Click a paper, or lasso/box-select some, to list them here.'),
    )
    selected = (
        meta[_picked]
        .with_columns(
            pl.Series(
                'island',
                [island_names.get(int(island_id[i]), 'open water') for i in _picked],
            )
        )
        .select('bibcode', 'first_author', 'year', 'title', 'citation_count', 'island')
        .sort('citation_count', descending=True)
    )
    table = mo.ui.table(selected, selection='single', page_size=15)
    table
    return (table,)


@app.cell
def _(con, mo, table):
    # One paper (single selection, or a click that selected exactly one): what it cites
    # and what cites it, within the corpus.
    _rows = table.value if len(table.value) else table.data
    mo.stop(len(_rows) != 1)
    _b = _rows['bibcode'][0]
    _q = (
        'SELECT p.bibcode, p.first_author, p.year, p.title, p.citation_count '
        'FROM citation_edges e JOIN papers p ON p.bibcode = e.{other} '
        'WHERE e.{me} = ? ORDER BY p.citation_count DESC NULLS LAST'
    )
    _refs = con.sql(
        _q.format(me='citing_bibcode', other='cited_bibcode'), params=[_b]
    ).pl()
    _cites = con.sql(
        _q.format(me='cited_bibcode', other='citing_bibcode'), params=[_b]
    ).pl()
    mo.vstack(
        [
            mo.md(f'### {_rows["title"][0]}\n`{_b}`'),
            mo.md(f'**References in corpus** ({len(_refs)})'),
            _refs,
            mo.md(f'**Cited by, in corpus** ({len(_cites)})'),
            _cites,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
