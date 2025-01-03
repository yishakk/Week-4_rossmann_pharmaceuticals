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
        cleaning_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])
        missing_cols = self.train.columns[self.train.isnull().any()]
        missing_cols = missing_cols.drop(['PromoInterval'])
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

    def analyze_seasonal_behavior(self):
        self.logger.info("Analyzing seasonal purchase behavior...")
        self.train['Month'] = self.train['Date'].dt.month
        monthly_sales = self.train.groupby('Month')['Sales'].mean()
        sns.barplot(x=monthly_sales.index, y=monthly_sales.values).set_title("Average Monthly Sales")
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

    def analyze_store_opening_behavior(self):
        self.logger.info("Analyzing customer behavior during store opening and closing times...")
        self.train['DayOfWeek'] = self.train['Date'].dt.dayofweek
        sns.boxplot(x='DayOfWeek', y='Sales', data=self.train).set_title("Sales by Day of the Week")
        plt.show()

    def analyze_weekday_weekend_sales(self):
        self.logger.info("Analyzing sales on weekdays vs weekends...")
        self.train['IsWeekend'] = self.train['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        weekend_sales = self.train.groupby('IsWeekend')['Sales'].mean()
        sns.barplot(x=weekend_sales.index, y=weekend_sales.values).set_title("Average Sales on Weekdays vs Weekends")
        plt.show()

    def analyze_assortment_impact(self):
        self.logger.info("Analyzing impact of assortment types on sales...")
        assortment_sales = self.train.groupby('Assortment')['Sales'].mean()
        sns.barplot(x=assortment_sales.index, y=assortment_sales.values).set_title("Average Sales by Assortment Type")
        plt.show()

    def analyze_competitor_distance(self):
        self.logger.info("Analyzing effect of competitor distance on sales...")
        sns.scatterplot(x='CompetitionDistance', y='Sales', data=self.train).set_title("Sales vs Competitor Distance")
        plt.show()

    def analyze_new_competitors(self):
        self.logger.info("Analyzing effect of new competitors...")
        self.train['HasNewCompetitor'] = self.train['CompetitionOpenSinceYear'].notnull()
        new_competitor_sales = self.train.groupby('HasNewCompetitor')['Sales'].mean()
        sns.barplot(x=new_competitor_sales.index, y=new_competitor_sales.values).set_title("Sales With/Without New Competitors")
        plt.show()

    def save_cleaned_data(self):
        self.logger.info("Saving cleaned datasets...")
        self.train.to_csv('../data/cleaned_train.csv', index=False)
        self.test.to_csv('../data/cleaned_test.csv', index=False)

    def run_analysis(self):
        self.load_data()
        self.clean_data()
        self.detect_outliers()
        self.analyze_promo_distribution()
        self.analyze_holiday_sales()
        self.analyze_seasonal_behavior()
        self.analyze_correlation()
        self.analyze_promo_impact()
        self.analyze_store_opening_behavior()
        self.analyze_weekday_weekend_sales()
        self.analyze_assortment_impact()
        self.analyze_competitor_distance()
        self.analyze_new_competitors()
        self.save_cleaned_data()