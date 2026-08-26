from pathlib import Path

import networkx as nx
import pandas as pd
from pyvis.network import Network


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "returnshield_labeled.csv"
)

RINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "detected_rings.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "graph"
    / "outputs"
)


def load_data():
    df = pd.read_csv(DATA_PATH)
    rings = pd.read_csv(RINGS_PATH)

    return df, rings


def get_top_ring(rings):

    top_ring = (
        rings
        .sort_values(
            "ring_risk_score",
            ascending=False,
        )
        .iloc[0]
    )

    members = (
        top_ring["members"]
        .split(",")
    )

    print(
        "Visualizing:",
        top_ring["detected_ring_id"],
    )

    print(
        "Risk score:",
        round(
            top_ring["ring_risk_score"],
            2,
        ),
    )

    print(
        "Members:",
        members,
    )

    return top_ring, members


def build_ring_graph(
    df,
    members,
):

    graph = nx.Graph()

    member_df = df[
        df["customer_id"].isin(
            members
        )
    ].copy()

    identity_columns = {
        "device_id": "device",
        "shipping_address_id": "address",
        "payment_instrument_id": "payment",
    }

    for customer in members:

        graph.add_node(
            customer,
            node_type="customer",
        )

    for column, node_type in identity_columns.items():

        temp = (
            member_df[
                [
                    "customer_id",
                    column,
                ]
            ]
            .drop_duplicates()
        )

        identity_counts = (
            temp.groupby(column)[
                "customer_id"
            ]
            .nunique()
        )

        shared_ids = (
            identity_counts[
                identity_counts > 1
            ]
            .index
        )

        temp = temp[
            temp[column].isin(
                shared_ids
            )
        ]

        for _, row in temp.iterrows():

            customer = row["customer_id"]
            identity = row[column]

            identity_node = (
                f"{node_type.upper()}::{identity}"
            )

            graph.add_node(
                identity_node,
                node_type=node_type,
                label=identity,
            )

            graph.add_edge(
                customer,
                identity_node,
                relationship=node_type,
            )

    return graph


def add_customer_information(
    graph,
    df,
    members,
):

    member_df = df[
        df["customer_id"].isin(
            members
        )
    ].copy()

    summary = (
        member_df
        .groupby("customer_id")
        .agg(
            total_orders=(
                "order_id",
                "nunique",
            ),
            returned_orders=(
                "is_returned",
                "sum",
            ),
            total_return_cost=(
                "return_cost",
                "sum",
            ),
            is_abuse=(
                "is_abuse",
                "max",
            ),
        )
    )

    summary["return_rate"] = (
        summary["returned_orders"]
        / summary["total_orders"]
    )

    for customer in members:

        row = summary.loc[
            customer
        ]

        graph.nodes[
            customer
        ][
            "return_rate"
        ] = float(
            row["return_rate"]
        )

        graph.nodes[
            customer
        ][
            "total_return_cost"
        ] = float(
            row["total_return_cost"]
        )

        graph.nodes[
            customer
        ][
            "is_abuse"
        ] = int(
            row["is_abuse"]
        )

    return graph


def create_visualization(
    graph,
    ring_id,
    risk_score,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{ring_id}.html"
    )

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#111111",
        font_color="white",
    )

    net.barnes_hut()

    for node, data in graph.nodes(
        data=True
    ):

        node_type = data.get(
            "node_type"
        )

        if node_type == "customer":

            return_rate = (
                data.get(
                    "return_rate",
                    0,
                )
            )

            return_cost = (
                data.get(
                    "total_return_cost",
                    0,
                )
            )

            title = (
                f"Customer: {node}<br>"
                f"Return rate: "
                f"{return_rate:.1%}<br>"
                f"Return value: "
                f"₹{return_cost:,.2f}"
            )

            net.add_node(
                node,
                label=node,
                title=title,
                shape="dot",
                size=28,
            )

        else:

            label = (
                data.get(
                    "label",
                    node,
                )
            )

            title = (
                f"{node_type.title()}: "
                f"{label}"
            )

            net.add_node(
                node,
                label=label,
                title=title,
                shape="box",
                size=20,
            )

    for source, target, data in graph.edges(
        data=True
    ):

        relationship = (
            data.get(
                "relationship",
                ""
            )
        )

        net.add_edge(
            source,
            target,
            title=relationship,
        )

    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "stabilization": {
              "iterations": 200
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )

    net.write_html(
        str(output_path)
    )

    print(
        "\nSaved visualization to:"
    )

    print(
        output_path
    )

    print(
        "\nRing risk score:",
        round(
            risk_score,
            2,
        ),
    )


def main():

    df, rings = load_data()

    top_ring, members = (
        get_top_ring(
            rings
        )
    )

    graph = build_ring_graph(
        df,
        members,
    )

    graph = add_customer_information(
        graph,
        df,
        members,
    )

    create_visualization(
        graph,
        top_ring[
            "detected_ring_id"
        ],
        top_ring[
            "ring_risk_score"
        ],
    )


if __name__ == "__main__":
    main()