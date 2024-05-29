"""
This is a boilerplate pipeline 'first_pipeline'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, pipeline, node
from alzheimers_prediction.pipelines.data_engineering.nodes import consolidate_data, process_gene_data, split_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=process_gene_data,
                inputs=['geneticValues',],
                outputs='gene',
                name="process_gene_data",
            ),
            node(
                func=consolidate_data,
                inputs=['gene','phenotipicValues'],
                outputs=['genetic_data', 'phenotypes_numeric'],
                name="consolidate_data",
            ),
            node(
                func=split_data,
                inputs=['genetic_data', 'phenotypes_numeric','params:split'],
                outputs=[ 'X_train', 'X_test', 'y_train', 'y_test'],
                name="split_data",
            ),
        ]
    )
