"""
This is a boilerplate pipeline 'export_results'
generated using Kedro 0.19.13
"""

from textwrap import dedent

import black
import pandas as pd
from jinja2 import Template

from cortex.modeling.chaid import ChaidTree
from cortex.modeling.dffilter import BaseDFFilter, IntervalDFFilter, TrivialDFFilter


def tree_to_python(tree_table: pd.DataFrame, parameters: dict) -> str:
    tree = ChaidTree(**parameters)
    tree.from_frame(tree_table)

    template = Template(
        dedent(
            """
            import pandas as pd
            from math import inf


            __all__ = ["eval_tree"]


            def eval_tree(data: pd.DataFrame | pd.Series) -> pd.Series:
                if isinstance(data, pd.Series):
                    return eval_node_0(data)

                return data.apply(eval_node_0, axis=1)
            {% for node in tree.nodes %}

            def eval_node_{{node.id}}(row: pd.Series) -> int:
                {%- if node.is_leaf %}
                return {{ node.id }}
                {% elif isinstance(node.children[0].dffilter, BaseDFFilter) %}
                {%- for child in node.children %}
                if row["{{child.dffilter.column}}"] in {{ set_to_str(child.dffilter.value) }}:
                    leaf_node, path, status = eval_node_{{child.id}}(row)
                    return leaf_node
                {% endfor %}
                return {{ node.id }}, [{{ node.id }}], False
                {% elif isinstance(node.children[0].dffilter, IntervalDFFilter) %}
                {%- for child in node.children %}
                if {{ child.dffilter.interval.lower }} <= row["{{ child.dffilter.column }}"] <= {{ child.dffilter.interval.upper }} :
                    leaf_node, path, status = eval_node_{{child.id}}(row)
                    return leaf_node
                {% endfor -%}
                return {{ node.id }}
                {%- endif %}
            {%- endfor -%}
            """
        )
    )

    def set_to_str(set_val):
        return "['" + "', '".join(set_val) + "']"

    return black.format_str(
        template.render(
            tree=tree,
            set_to_str=set_to_str,
            isinstance=isinstance,
            BaseDFFilter=BaseDFFilter,
            IntervalDFFilter=IntervalDFFilter,
            encoder=encoder,
        ),
        mode=black.FileMode(),
    )


def tree_to_lookup_table(tree: ChaidTree) -> pd.DataFrame:
    output = []
    for node in tree.nodes:
        if isinstance(node.dffilter, TrivialDFFilter):
            output.append((node.id, "", ""))
            continue

        if isinstance(node.dffilter, BaseDFFilter):
            output.append(
                (
                    node.depth,
                    node.id,
                    node.dffilter.column,
                    "\n".join(node.dffilter.value),
                )
            )
            continue

        if isinstance(node.dffilter, IntervalDFFilter):
            output.append(
                (
                    node.depth,
                    node.id,
                    node.dffilter.column,
                    str(node.dffilter.interval),
                )
            )
            continue

    return pd.DataFrame(output, columns=["Level", "Node", "Column", "Value"])
