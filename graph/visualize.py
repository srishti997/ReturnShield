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


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    df = pd.read_csv(
        DATA_PATH
    )

    rings = pd.read_csv(
        RINGS_PATH
    )

    return df, rings


# =========================================================
# CLEAN MEMBER LIST
# =========================================================

def parse_members(
    members_value,
):

    members = [
        member.strip()
        for member in str(
            members_value
        ).split(",")
        if member.strip()
    ]

    return members


# =========================================================
# BUILD GRAPH FOR ONE RING
# =========================================================

def build_ring_graph(
    df,
    members,
):

    graph = nx.Graph()

    member_df = df[
        df[
            "customer_id"
        ].isin(
            members
        )
    ].copy()

    identity_columns = {
        "device_id": "device",
        "shipping_address_id": "address",
        "payment_instrument_id": "payment",
    }


    # -----------------------------------------------------
    # ADD CUSTOMER NODES
    # -----------------------------------------------------

    for customer in members:

        graph.add_node(
            customer,
            node_type="customer",
        )


    # -----------------------------------------------------
    # ADD SHARED IDENTITY NODES
    # -----------------------------------------------------

    for (
        column,
        node_type,
    ) in identity_columns.items():

        if column not in member_df.columns:

            continue

        temp = (
            member_df[
                [
                    "customer_id",
                    column,
                ]
            ]
            .dropna()
            .drop_duplicates()
        )

        identity_counts = (
            temp.groupby(
                column
            )[
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
            temp[
                column
            ].isin(
                shared_ids
            )
        ]

        for _, row in temp.iterrows():

            customer = (
                row[
                    "customer_id"
                ]
            )

            identity = str(
                row[
                    column
                ]
            )

            identity_node = (
                f"{node_type.upper()}"
                f"::{identity}"
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


# =========================================================
# ADD CUSTOMER INFORMATION
# =========================================================

def add_customer_information(
    graph,
    df,
    members,
):

    member_df = df[
        df[
            "customer_id"
        ].isin(
            members
        )
    ].copy()

    summary = (
        member_df
        .groupby(
            "customer_id"
        )
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

    summary[
        "return_rate"
    ] = (
        summary[
            "returned_orders"
        ]
        / summary[
            "total_orders"
        ]
    )


    for customer in members:

        if (
            customer
            not in summary.index
        ):

            continue

        row = summary.loc[
            customer
        ]

        graph.nodes[
            customer
        ][
            "return_rate"
        ] = float(
            row[
                "return_rate"
            ]
        )

        graph.nodes[
            customer
        ][
            "total_return_cost"
        ] = float(
            row[
                "total_return_cost"
            ]
        )

        graph.nodes[
            customer
        ][
            "is_abuse"
        ] = int(
            row[
                "is_abuse"
            ]
        )

    return graph


# =========================================================
# CREATE VISUALIZATION FOR ONE RING
# =========================================================

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


    # -----------------------------------------------------
    # ADD NODES
    # -----------------------------------------------------

    for node, data in graph.nodes(
        data=True
    ):

        node_type = data.get(
            "node_type"
        )


        # -------------------------------------------------
        # CUSTOMER NODE
        # -------------------------------------------------

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

            is_abuse = (
                data.get(
                    "is_abuse",
                    0,
                )
            )

            title = (
                f"Customer: {node}<br>"
                f"Return rate: "
                f"{return_rate:.1%}<br>"
                f"Return value: "
                f"₹{return_cost:,.2f}<br>"
                f"Known abuse label: "
                f"{is_abuse}"
            )

            net.add_node(
                node,
                label=node,
                title=title,
                shape="dot",
                size=28,
            )


        # -------------------------------------------------
        # IDENTITY NODE
        # -------------------------------------------------

        else:

            label = data.get(
                "label",
                node,
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


    # -----------------------------------------------------
    # ADD EDGES
    # -----------------------------------------------------

    for (
        source,
        target,
        data,
    ) in graph.edges(
        data=True
    ):

        relationship = (
            data.get(
                "relationship",
                "",
            )
        )

        net.add_edge(
            source,
            target,
            title=relationship,
        )


    # -----------------------------------------------------
    # NETWORK OPTIONS
    # -----------------------------------------------------

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
        str(
            output_path
        )
    )

    print(
        f"Saved {ring_id} "
        f"(risk={risk_score:.2f}) "
        f"-> {output_path}"
    )

    return output_path


# =========================================================
# GENERATE ALL RING VISUALIZATIONS
# =========================================================

def generate_all_ring_visualizations(
    df,
    rings,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = 0
    skipped = 0

    rings = rings.sort_values(
        "ring_risk_score",
        ascending=False,
    )

    print()
    print(
        "Generating ReturnShield "
        "ring visualizations..."
    )
    print(
        f"Detected rings: {len(rings)}"
    )
    print()


    for _, ring in rings.iterrows():

        ring_id = str(
            ring[
                "detected_ring_id"
            ]
        )

        risk_score = float(
            ring[
                "ring_risk_score"
            ]
        )

        members = parse_members(
            ring[
                "members"
            ]
        )


        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if not members:

            print(
                f"Skipping {ring_id}: "
                "no members found."
            )

            skipped += 1

            continue


        try:

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
                ring_id,
                risk_score,
            )

            generated += 1


        except Exception as error:

            print(
                f"Could not generate "
                f"{ring_id}: {error}"
            )

            skipped += 1


    print()
    print(
        "================================="
    )
    print(
        "GRAPH GENERATION COMPLETE"
    )
    print(
        "================================="
    )
    print(
        f"Generated: {generated}"
    )
    print(
        f"Skipped:   {skipped}"
    )
    print(
        f"Output:    {OUTPUT_DIR}"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    df, rings = load_data()

    generate_all_ring_visualizations(
        df,
        rings,
    )


if __name__ == "__main__":

    main()