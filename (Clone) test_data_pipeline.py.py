# Databricks notebook source
# MAGIC %pip install pytest
# MAGIC

# COMMAND ----------

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, mean

# Initialize PySpark Test Session
@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local").appName("TestANZDataPipeline").getOrCreate()

# COMMAND ----------

# Sample Data for Testing
@pytest.fixture
def sample_data(spark):
    data = [
        ("debit", "M", 30, 1, 100.0, "2024-01-10", "Amazon"),
        ("credit", "F", 25, 0, 200.0, "2024-01-11", "Walmart"),
        ("debit", "F", 35, 1, 50.0, "2024-01-12", "Target"),
        ("credit", "M", 40, 0, 500.0, "2024-01-13", "Apple"),
        (None, "F", 28, 1, 150.0, "2024-01-14", "Nike")  # Edge case with missing movement
    ]
    columns = ["movement", "gender", "age", "card_present_flag", "amount", "transaction_date", "merchant_name"]
    return spark.createDataFrame(data, columns)

# COMMAND ----------

#  1. Test for Null Values
def test_null_values(sample_data):
    null_count = sample_data.filter(sample_data.movement.isNull()).count()
    assert null_count == 1, f"Expected 1 null value in 'movement', but found {null_count}"
    print("Null values test passed")

# COMMAND ----------

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, sum, mean

# Initialize PySpark Test Session
@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder.master("local").appName("TestANZDataPipeline").getOrCreate()

# Sample Data for Testing
@pytest.fixture
def sample_data(spark):
    data = [
        ("debit", "M", 30, 1, 100.0, "2024-01-10", "Amazon"),
        ("credit", "F", 25, 0, 200.0, "2024-01-11", "Walmart"),
        ("debit", "F", 35, 1, 50.0, "2024-01-12", "Target"),
        ("credit", "M", 40, 0, 500.0, "2024-01-13", "Apple"),
        (None, "F", 28, 1, 150.0, "2024-01-14", "Nike")  # Edge case with missing movement
    ]
    columns = ["movement", "gender", "age", "card_present_flag", "amount", "transaction_date", "merchant_name"]
    return spark.createDataFrame(data, columns)

#  1. Test for Null Values
def test_null_values(sample_data):
    null_count = sample_data.filter(sample_data.movement.isNull()).count()
    assert null_count == 1, f"Expected 1 null value in 'movement', but found {null_count}"
    print("Null values test passed")

#  2. Test for Transaction Aggregation
def test_transaction_aggregation(sample_data):
    result = sample_data.groupBy("movement").agg(sum("amount").alias("total_amount"))
    result_dict = {row["movement"]: row["total_amount"] for row in result.collect()}
    
    assert result_dict["debit"] == 150.0, "Debit transactions total amount mismatch"
    assert result_dict["credit"] == 700.0, "Credit transactions total amount mismatch"
    print("Transaction aggregation test passed")

#  3. Test for Average Transaction Amount
def test_average_transaction(sample_data):
    result = sample_data.groupBy("movement").agg(mean("amount").alias("avg_amount"))
    avg_dict = {row["movement"]: row["avg_amount"] for row in result.collect()}
    
    assert round(avg_dict["debit"], 2) == 75.0, "Debit avg amount incorrect"
    assert round(avg_dict["credit"], 2) == 350.0, "Credit avg amount incorrect"
    print("Average transaction amount test passed")

#  4. Test for Card Presence Flag Handling (NULLs replaced with 0)
def test_card_present_flag(sample_data):
    clean_data = sample_data.fillna({"card_present_flag": 0})
    null_count = clean_data.filter(clean_data.card_present_flag.isNull()).count()
    
    assert null_count == 0, "Null values in 'card_present_flag' should be replaced with 0"
    print("Card presence flag test passed")

# 5. Test for High-Value Transactions Detection
def test_high_value_transactions(sample_data):
    high_value = sample_data.orderBy(sample_data.amount.desc()).limit(1)
    assert high_value.collect()[0]["amount"] == 500.0, "Top transaction value mismatch"
    print("High value transactions test passed")

#  6. Test for Correct Data Types
def test_data_types(sample_data):
    expected_schema = {
        "movement": "string",
        "gender": "string",
        "age": "int",
        "card_present_flag": "int",
        "amount": "double",
        "transaction_date": "string",
        "merchant_name": "string"
    }
    
    for field in sample_data.schema.fields:
        assert field.dataType.simpleString() == expected_schema[field.name], f"Incorrect datatype for {field.name}"
    print("Data types test passed")

#  7. Test for Missing Transactions Filtering
def test_missing_transactions(sample_data):
    valid_data = sample_data.filter(sample_data.movement.isNotNull())
    assert valid_data.count() == 4, "Missing transactions not handled correctly"

    print("Missing transactions test passed")


# COMMAND ----------


