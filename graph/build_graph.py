from pathlib import Path

import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "returnshield_labeled.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "graph_edges.csv"
)


def load_data():
    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded dataset: "
        f"{len(df)} orders"
    )

    return df


def build_graph(df):

    graph = nx.Graph()

    # One row per unique customer identity relationship
    identity_df = (
        df[
            [
                "customer_id",
                "device_id",
                "shipping_address_id",
                "payment_instrument_id",
                "ip_address",
            ]
        ]
        .drop_duplicates()
    )

    for _, row in identity_df.iterrows():

        customer = f"CUSTOMER::{row['customer_id']}"

        graph.add_node(
            customer,
            node_type="customer",
            raw_id=row["customer_id"],
        )

        identities = [
            (
                f"DEVICE::{row['device_id']}",
                "device",
            ),
            (
                f"ADDRESS::{row['shipping_address_id']}",
                "address",
            ),
            (
                f"PAYMENT::{row['payment_instrument_id']}",
                "payment",
            ),
            (
                f"IP::{row['ip_address']}",
                "ip",
            ),
        ]

        for identity_node, node_type in identities:

            graph.add_node(
                identity_node,
                node_type=node_type,
            )

            graph.add_edge(
                customer,
                identity_node,
                relationship=node_type,
            )

    return graph


def graph_summary(graph):

    print("\n--- GRAPH SUMMARY ---")

    print(
        "Total nodes:",
        graph.number_of_nodes(),
    )

    print(
        "Total edges:",
        graph.number_of_edges(),
    )

    node_types = {}

    for _, data in graph.nodes(data=True):

        node_type = data.get(
            "node_type",
            "unknown",
        )

        node_types[node_type] = (
            node_types.get(
                node_type,
                0,
            )
            + 1
        )

    print("\nNode types:")

    for key, value in node_types.items():
        print(
            f"{key}: {value}"
        )


def export_edges(graph):

    rows = []

    for source, target, data in graph.edges(data=True):

        rows.append(
            {
                "source": source,
                "target": target,
                "relationship":
                    data.get(
                        "relationship",
                        "",
                    ),
            }
        )

    edge_df = pd.DataFrame(rows)

    edge_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nSaved graph edges to:"
    )

    print(
        OUTPUT_PATH
    )


def main():

    df = load_data()

    graph = build_graph(df)

    graph_summary(graph)

    export_edges(graph)


if __name__ == "__main__":
    main()