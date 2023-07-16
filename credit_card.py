from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class CreditCard(BaseModel):
    Customer_Age: int
    Total_Relationship_Count: int
    Months_Inactive_12_mon: int
    Contacts_Count_12_mon: int
    Credit_Limit: int
    Total_Revolving_Bal: int
    Avg_Open_To_Buy: int
    Total_Amt_Chng_Q4_Q1: float
    Avg_Utilization_Ratio: float
    Blue: int
    Gold: int
    Platinum: int
    Silver: int