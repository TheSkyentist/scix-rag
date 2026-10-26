import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Browse the corpus

    Read-only view of `data/papers.duckdb`. Pick a table and page through it, or write
    your own SQL in the last cell.
    """)
    return


@app.cell
def _():
    import duckdb
    import marimo as mo

    return duckdb, mo


@app.cell
def _(duckdb, mo):
    db_path = mo.notebook_dir().parent / 'data' / 'papers.duckdb'
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
def _(con, mo):
    tables = [
        r[0]
        for r in con.sql(
            'SELECT table_name FROM information_schema.tables '
            "WHERE table_schema = 'main' ORDER BY 1"
        ).fetchall()
    ]
    table = mo.ui.dropdown(
        tables, value='papers' if 'papers' in tables else tables[0], label='Table'
    )
    page_size = mo.ui.dropdown([25, 50, 100, 500], value=100, label='Rows per page')
    mo.hstack([table, page_size], justify='start')
    return page_size, table


@app.cell
def _(con, mo, page_size, table):
    n_rows = con.sql(f'SELECT count(*) FROM "{table.value}"').fetchall()[0][0]
    n_pages = max(1, -(-n_rows // page_size.value))
    page = mo.ui.number(1, n_pages, value=1, label=f'Page (of {n_pages:,})')
    mo.hstack([page, mo.md(f'{n_rows:,} rows')], justify='start', align='center')
    return (page,)


@app.cell
def _(con, mo, page, page_size, table):
    # Vectors are unreadable in a grid, so the embedding column is left out.
    columns = [
        r[0]
        for r in con.sql(f'DESCRIBE "{table.value}"').fetchall()
        if r[0] != 'embedding'
    ]
    select = ', '.join(f'"{c}"' for c in columns)
    offset = (page.value - 1) * page_size.value
    rows = con.sql(
        f'SELECT {select} FROM "{table.value}" LIMIT {page_size.value} OFFSET {offset}'
    ).pl()
    mo.ui.table(rows, page_size=page_size.value, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Your own SQL

    Edit the query below. Vectors live in `chunk_embeddings.embedding`; leave that column
    out (`SELECT * EXCLUDE (embedding)`) unless you want 1,024 numbers per row.
    """)
    return


@app.cell
def _(con, mo):
    _df = mo.sql(
        f"""
        SELECT first_author, year, title, citation_count
        FROM papers
        ORDER BY citation_count DESC NULLS LAST
        LIMIT 100
        """,
        engine=con
    )
    return


if __name__ == "__main__":
    app.run()
