from kedro.pipeline import Pipeline, pipeline,node
from alzheimers_prediction.pipelines.data_science.nodes import train_model,test_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
                func=train_model,
                inputs=["X_train", "y_train", "params:models.random_forest.name", "params:models.random_forest.search_space"],
                outputs="rf_model",
                name="train_rf_model_node",
            ),
            node(
                func=test_model,
                inputs=['rf_model','X_test','y_test'],
                outputs=['metrics','rf_model_best'],
                name="test_model",
            ),
            ])


