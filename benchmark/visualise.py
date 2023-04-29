import altair as alt
import pandas as pd

if __name__ == "__main__":
    df = pd.read_json("assets/results.jsonl", lines=True)

    df["num_embeddings_formatted"] = df["num_embeddings"].apply(lambda x: f"{x:,}")
    df = df[df["num_embeddings"] != 1_500_000]
    df["build_time_sec"] = df["build_time"] / 1000

    query_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("index", title=None),
            y=alt.Y("mean", title="Time (ms)"),
            color=alt.Color("index:O", scale=alt.Scale(scheme="set1"), title="Index"),
        )
        .facet(
            column=alt.Column(
                "num_embeddings_formatted:O",
                header=alt.Header(title=None),
                sort=alt.EncodingSortField("num_embeddings"),
            ),
        )
        .properties(title="Query time for k=10 (ms)")
    )

    df2 = df.sample(frac=1)
    build_chart = (
        alt.Chart(df2)
        .mark_bar()
        .encode(
            x=alt.X("index", title=None),
            y=alt.Y("build_time_sec", title="Time (s)"),
            color=alt.Color("index:O", scale=alt.Scale(scheme="set1"), title="Index"),
        )
        .facet(
            column=alt.Column(
                "num_embeddings_formatted:O",
                header=alt.Header(title=None),
                sort=alt.EncodingSortField("num_embeddings"),
            ),
        )
        .properties(title="Build time (s)")
    )

    (query_chart | build_chart).save("assets/fig_benchmark.png")
