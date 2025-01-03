import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class CustomerBehaviorEDA:
    def __init__(self, train_path, test_path, store_path):
        self.train_path = train_path
        self.test_path = test_path
        self.store_path = store_path
        self.train = None
        self.test = None
        self.store = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)

    def load_data(self):
        self.logger.info("Loading datasets...")
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        self.store = pd.read_csv(self.store_path)
        self.train = self.train.merge(self.store, on='Store', how='left')
        self.test = self.test.merge(self.store, on='Store', how='left')

    def clean_data(self):
        self.logger.info("Cleaning data...")
        cleaning_pipeline = self.train.select_dtypes(include=['number']).columns
        cleaning_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])
        missing_cols = self.train.columns[self.train.isnull().any()]
        self.logger.info(f"Columns with missing values: {missing_cols}")
        self.train[missing_cols] = cleaning_pipeline.fit_transform(self.train[missing_cols])
        self.test[missing_cols] = cleaning_pipeline.transform(self.test[missing_cols])

    def detect_outliers(self):
        self.logger.info("Detecting outliers...")
        for col in ['Sales', 'Customers']:
            z_scores = (self.train[col] - self.train[col].mean()) / self.train[col].std()
            outliers = self.train[z_scores.abs() > 3]
            self.logger.info(f"Found {len(outliers)} outliers in {col}")
            self.train = self.train[z_scores.abs() <= 3]

    def analyze_promo_distribution(self):
        self.logger.info("Analyzing promo distributions...")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.histplot(self.train['Promo'], ax=ax[0]).set_title("Train Promo Distribution")
        sns.histplot(self.test['Promo'], ax=ax[1]).set_title("Test Promo Distribution")
        plt.show()

    def analyze_holiday_sales(self):
        self.logger.info("Analyzing sales behavior around holidays...")
        self.train['Date'] = pd.to_datetime(self.train['Date'])
        self.train['HolidayPeriod'] = self.train['StateHoliday'].apply(lambda x: x if x != '0' else 'No Holiday')
        sns.lineplot(data=self.train, x='Date', y='Sales', hue='HolidayPeriod').set_title("Sales Around Holidays")
        plt.show()

    def analyze_correlation(self):
        self.logger.info("Analyzing correlation between sales and customers...")
        correlation = self.train[['Sales', 'Customers']].corr()
        self.logger.info(f"Correlation matrix:\n{correlation}")
        sns.heatmap(correlation, annot=True, cmap='coolwarm').set_title("Correlation Between Sales and Customers")
        plt.show()

    def analyze_promo_impact(self):
        self.logger.info("Analyzing impact of promos on sales and customers...")
        promo_sales = self.train.groupby('Promo')['Sales'].mean()
        promo_customers = self.train.groupby('Promo')['Customers'].mean()
        self.logger.info(f"Average sales with promo:\n{promo_sales}")
        self.logger.info(f"Average customers with promo:\n{promo_customers}")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.barplot(x=promo_sales.index, y=promo_sales.values, ax=ax[0]).set_title("Average Sales With/Without Promo")
        sns.barplot(x=promo_customers.index, y=promo_customers.values, ax=ax[1]).set_title("Average Customers With/Without Promo")
        plt.show()

    def save_cleaned_data(self):
        self.logger.info("Saving cleaned datasets...")
        self.train.to_csv('cleaned_train.csv', index=False)
        self.test.to_csv('cleaned_test.csv', index=False)

    def run_analysis(self):
        self.load_data()
        self.clean_data()
        self.detect_outliers()
        self.analyze_promo_distribution()
        self.analyze_holiday_sales()
        self.analyze_correlation()
        self.analyze_promo_impact()
        self.save_cleaned_data()
