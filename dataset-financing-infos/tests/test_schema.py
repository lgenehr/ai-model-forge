from src.pipeline.schema import DatasetRow, DatasetMeta

def test_dataset_row_valid():
    row = DatasetRow(
        instruction="Tell me about X",
        input="",
        output="X is Y",
        meta=DatasetMeta(
            source="test",
            date="2025-01-01",
            topics=["tech"]
        )
    )
    data = row.model_dump()
    assert data["instruction"] == "Tell me about X"
    assert data["meta"]["date"] == "2025-01-01"
