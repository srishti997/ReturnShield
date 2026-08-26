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
    / "detected_rings.csv"
)


def load_data():
    df = pd.read_csv(DATA_PATH)

    print(
        f"Loaded dataset: {len(df)} orders"
    )

    return df


# ---------------------------------------------------------
# BUILD CUSTOMER RELATIONSHIP GRAPH
# ---------------------------------------------------------

def build_customer_graph(df):

    graph = nx.Graph()

    customers = df[
        "customer_id"
    ].unique()

    for customer in customers:
        graph.add_node(customer)

    # -----------------------------------------------------
    # Connect customers that share identities
    # -----------------------------------------------------

    identity_columns = [
        "device_id",
        "shipping_address_id",
        "payment_instrument_id",
    ]

    for column in identity_columns:

        identity_groups = (
            df[
                [
                    "customer_id",
                    column,
                ]
            ]
            .drop_duplicates()
            .groupby(column)[
                "customer_id"
            ]
            .apply(list)
        )

        for identity, members in identity_groups.items():

            # One customer using an identity
            # creates no relationship.
            if len(members) < 2:
                continue

            # Connect every pair of customers
            # sharing the identity.
            for i in range(len(members)):

                for j in range(
                    i + 1,
                    len(members),
                ):

                    customer_a = members[i]
                    customer_b = members[j]

                    if graph.has_edge(
                        customer_a,
                        customer_b,
                    ):

                        graph[
                            customer_a
                        ][
                            customer_b
                        ][
                            "shared_identities"
                        ].append(
                            {
                                "type": column,
                                "value": identity,
                            }
                        )

                    else:

                        graph.add_edge(
                            customer_a,
                            customer_b,
                            shared_identities=[
                                {
                                    "type": column,
                                    "value": identity,
                                }
                            ],
                        )

    return graph


# ---------------------------------------------------------
# CUSTOMER BEHAVIOUR SUMMARY
# ---------------------------------------------------------

def build_customer_summary(df):

    summary = (
        df.groupby("customer_id")
        .agg(
            total_orders=(
                "order_id",
                "nunique",
            ),
            returned_orders=(
                "is_returned",
                "sum",
            ),
            total_spend=(
                "order_value",
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
            abuse_type=(
                "abuse_type",
                "first",
            ),
        )
        .reset_index()
    )

    summary["return_rate"] = (
        summary["returned_orders"]
        / summary["total_orders"]
    )

    return summary.set_index(
        "customer_id"
    )


# ---------------------------------------------------------
# DETECT CONNECTED GROUPS
# ---------------------------------------------------------

def detect_connected_groups(
    graph,
    summary,
):

    results = []

    connected_components = list(
        nx.connected_components(
            graph
        )
    )

    ring_number = 1

    for component in connected_components:

        # Single users are not rings.
        if len(component) < 2:
            continue

        subgraph = graph.subgraph(
            component
        )

        members = list(component)

        member_summary = (
            summary.loc[members]
        )

        avg_return_rate = (
            member_summary[
                "return_rate"
            ].mean()
        )

        total_return_value = (
            member_summary[
                "total_return_cost"
            ].sum()
        )

        abusive_members = int(
            member_summary[
                "is_abuse"
            ].sum()
        )

        # Number of relationship edges.
        edge_count = (
            subgraph.number_of_edges()
        )

        # Density tells us how interconnected
        # the accounts are.
        density = nx.density(
            subgraph
        )

        # Count the identity connections.
        device_links = 0
        address_links = 0
        payment_links = 0

        for _, _, data in subgraph.edges(
            data=True
        ):

            relationships = data.get(
                "shared_identities",
                [],
            )

            for relationship in relationships:

                relationship_type = (
                    relationship["type"]
                )

                if (
                    relationship_type
                    == "device_id"
                ):
                    device_links += 1

                elif (
                    relationship_type
                    == "shipping_address_id"
                ):
                    address_links += 1

                elif (
                    relationship_type
                    == "payment_instrument_id"
                ):
                    payment_links += 1

        results.append(
            {
                "detected_ring_id":
                    f"DETECTED_{ring_number:03d}",

                "member_count":
                    len(members),

                "members":
                    ",".join(
                        sorted(members)
                    ),

                "relationship_edges":
                    edge_count,

                "device_links":
                    device_links,

                "address_links":
                    address_links,

                "payment_links":
                    payment_links,

                "network_density":
                    density,

                "avg_return_rate":
                    avg_return_rate,

                "total_return_value":
                    total_return_value,

                # Ground truth is included only
                # for evaluation of our prototype.
                "known_abusive_members":
                    abusive_members,
            }
        )

        ring_number += 1

    return pd.DataFrame(
        results
    )


# ---------------------------------------------------------
# SIMPLE RISK SCORE
# ---------------------------------------------------------

def add_ring_risk_score(rings):

    if rings.empty:
        return rings

    # This is intentionally interpretable.
    #
    # We are NOT training another ML model here.
    # This is a graph/rules risk score.

    rings["relationship_score"] = (
        rings["device_links"]
        + rings["address_links"]
        + rings["payment_links"]
    )

    rings["ring_risk_score"] = (
        0.35
        * rings[
            "avg_return_rate"
        ].clip(0, 1)
        +
        0.25
        * rings[
            "network_density"
        ].clip(0, 1)
        +
        0.20
        * (
            rings[
                "device_links"
            ]
            / rings[
                "member_count"
            ]
        ).clip(0, 1)
        +
        0.10
        * (
            rings[
                "address_links"
            ]
            / rings[
                "member_count"
            ]
        ).clip(0, 1)
        +
        0.10
        * (
            rings[
                "payment_links"
            ]
            / rings[
                "member_count"
            ]
        ).clip(0, 1)
    )

    rings[
        "ring_risk_score"
    ] = (
        rings[
            "ring_risk_score"
        ]
        * 100
    )

    return rings


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

def print_summary(rings):

    print(
        "\n--- RING DETECTION SUMMARY ---"
    )

    print(
        "Connected groups found:",
        len(rings),
    )

    if rings.empty:
        return

    print(
        "Customers in connected groups:",
        rings[
            "member_count"
        ].sum(),
    )

    print(
        "\nTop suspicious groups:"
    )

    display_columns = [
        "detected_ring_id",
        "member_count",
        "device_links",
        "address_links",
        "payment_links",
        "avg_return_rate",
        "network_density",
        "ring_risk_score",
        "known_abusive_members",
    ]

    print(
        rings[
            display_columns
        ]
        .sort_values(
            "ring_risk_score",
            ascending=False,
        )
        .head(10)
        .to_string(
            index=False
        )
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    df = load_data()

    graph = build_customer_graph(
        df
    )

    summary = build_customer_summary(
        df
    )

    print(
        "\nCustomer graph:"
    )

    print(
        "Nodes:",
        graph.number_of_nodes()
    )

    print(
        "Edges:",
        graph.number_of_edges()
    )

    rings = detect_connected_groups(
        graph,
        summary,
    )

    rings = add_ring_risk_score(
        rings
    )

    rings = rings.sort_values(
        "ring_risk_score",
        ascending=False,
    )

    rings.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(
        rings
    )

    print(
        "\nSaved detected rings to:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()