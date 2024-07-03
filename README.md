<table style="border-collapse: collapse;">
  <tr>
    <td style="text-align: left; border: none;">
      <h1 style="font-weight: bold;">California Real Estate Price Prediction</h1>
      <p>Using real estate listings collected in the first 6 months in 2021</p>
    </td>
    <td style="border: none;">
      <img src="images/dataset-cover.jpg" alt="Dataset Cover">
    </td>
  </tr>
</table>

## Introduction
This is a capstone project submission for UC Berkeley School of Engineering and Haas School of Business' certification program in Machine Learning. 
### Topic: 
Using the real estate sales dataset for houses in California for 2021 January through July, identify a model and key features that can help predict prices for single family homes.
### Author: 
Vinay Jain [LinkedIn Profile](https://www.linkedin.com/in/vinay-jain-5151ba/), [Email Vinay](mailto:vinay.jn@gmail.com)

## Data set information:
#### Source:
- Dataset is from Kaggle: https://www.kaggle.com/datasets/yellowj4acket/real-estate-california/data 
- This dataset shows real estate listing for California (US) for the first 7 months of 2021. Prices are listed in USD.

#### Data inderstanding:
<pre>
Data Set Characteristics:  Multivariate
Area: Real Estate/Housing
Attribute Characteristics: Real
Missing Values? None
</pre>

#### Attribute information:
There is one CSV file: RealEstate_California.csv.

There are a total of 35,389 records with 39 columns.
<pre>
 #   Column              Non-Null Count  Dtype  
---  ------              --------------  -----  
 0   sequence            35389 non-null  int64  
 1   id                  35389 non-null  object 
 2   stateId             35389 non-null  int64  
 3   countyId            35389 non-null  int64  
 4   cityId              35389 non-null  int64  
 5   country             35389 non-null  object 
 6   datePostedString    35386 non-null  object 
 7   is_bankOwned        35389 non-null  int64  
 8   is_forAuction       35389 non-null  int64  
 9   event               35100 non-null  object 
 10  time                35100 non-null  float64
 11  price               35389 non-null  float64
 12  pricePerSquareFoot  35389 non-null  float64
 13  city                35389 non-null  object 
 14  state               35389 non-null  object 
 15  yearBuilt           35389 non-null  int64  
 16  streetAddress       35388 non-null  object 
 17  zipcode             35364 non-null  float64
 18  longitude           35389 non-null  float64
 19  latitude            35389 non-null  float64
 20  hasBadGeocode       35389 non-null  int64  
 21  description         35110 non-null  object 
 22  currency            35389 non-null  object 
 23  livingArea          35389 non-null  float64
 24  livingAreaValue     35389 non-null  float64
 25  lotAreaUnits        35389 non-null  object 
 26  bathrooms           35389 non-null  float64
 27  bedrooms            35389 non-null  float64
 28  buildingArea        35389 non-null  float64
 29  parking             35389 non-null  int64  
 30  garageSpaces        35389 non-null  float64
 31  hasGarage           35389 non-null  int64  
 32  levels              35389 non-null  object 
 33  pool                35389 non-null  int64  
 34  spa                 35389 non-null  int64  
 35  isNewConstruction   35389 non-null  int64  
 36  hasPetsAllowed      35389 non-null  int64  
 37  homeType            35389 non-null  object 
 38  county              35389 non-null  object 
dtypes: float64(12), int64(14), object(13)
memory usage: 10.5+ MB
</pre>

## Data analysis:
Extensive data analysis was conducted for all the 39 column. All of the data analysis can be found in this file: california_real_estate_price_predictor_data_analysis.ipynb

The final snapshot of the analysis is summarized in the table below:
<pre>
 0   sequence            35389 non-null  int64    x   Drop column -- its a row ID column; not useful for analysis
 1   id                  35389 non-null  object   x   Drop column -- its a row ID column; not useful for analysis
 2   stateId             35389 non-null  int64    x   Drop column -- all of the data is for only state (California)
 3   countyId            35389 non-null  int64    x   Drop column -- drop in favor of zipcode
 4   cityId              35389 non-null  int64    x   Drop column -- drop in favor of zipcode
 5   country             35389 non-null  object   x   Drop column -- all of the data is for only country  (US) 
 6   datePostedString    35386 non-null  object   x   Drop all records older than '2021-01-01', then drop this column since there won't be a need to do time series analysis (data spread over only 7 months); Create a new 'month' column as it might help with modeling
 7   is_bankOwned        35389 non-null  int64    x   Drop records where value is 1 then drop this column
 8   is_forAuction       35389 non-null  int64    x   Drop records where value is 1 then drop this column
 9   event               35100 non-null  object   x   Drop records where 'event'='Listing removed' OR 'event'='Listed for rent' (interested in sale related data only)
 10  time                35100 non-null  float64  x   Drop column -- not relevant for this analysis
 11  price               35389 non-null  float64  x   Drop records where price=0 -- price is the target variable. Need this value for processing.
 12  pricePerSquareFoot  35389 non-null  float64  x   Drop this column because of a very weak correlation with price.
 13  city                35389 non-null  object   x   Drop column -- drop in favor of cityId
 14  state               35389 non-null  object   x   Drop column -- drop in favor of stateId
 15  yearBuilt           35389 non-null  int64    x   Convert to categorical column
 16  streetAddress       35388 non-null  object   x   Drop column -- too unique to be used for modeling
 17  zipcode             35364 non-null  float64  x   Convert to categorical column
 18  longitude           35389 non-null  float64  x   Drop column -- drop in favor of zipcode
 19  latitude            35389 non-null  float64  x   Drop column -- drop in favor of zipcode
 20  hasBadGeocode       35389 non-null  int64    x   Drop column -- no records found after other cleanups that have a value of 1
 21  description         35110 non-null  object   x   Retain as-is -- will be used for NLP analysis
 22  currency            35389 non-null  object   x   Drop column -- all of the data is for only one currency (USD)
 23  livingArea          35389 non-null  float64  x   Drop column -- drop in favor of livingAreaValue (same values)
 24  livingAreaValue     35389 non-null  float64  x   Drop records where livingAreaValue=0
 25  lotAreaUnits        35389 non-null  object   x   Drop column -- no corresponding lotArea found in the dataset so units are not useful
 26  bathrooms           35389 non-null  float64  x   Drop records where bathrooms=0
 27  bedrooms            35389 non-null  float64  x   Drop records where bedrooms=0
 28  buildingArea        35389 non-null  float64  x   Drop column -- almost 16K out of the total valid 19.7K total records have a value of 0, hence not useful
 29  parking             35389 non-null  int64    x   Convert to bool
 30  garageSpaces        35389 non-null  float64  x   Retain records that have (df['garageSpaces'] > 0) | (df['hasGarage'] == 0) and  (df['garageSpaces'] == 0) | (df['hasGarage'] > 0), then drop 'hasGarage' column
 31  hasGarage           35389 non-null  int64    x   Drop column in favor of garageSpaces since this is just a binary column
 32  levels              35389 non-null  object   x   Consolidate all the possible values into 3 values
 33  pool                35389 non-null  int64    x   Convert to bool
 34  spa                 35389 non-null  int64    x   Convert to bool
 35  isNewConstruction   35389 non-null  int64    x   Convert to bool 
 36  hasPetsAllowed      35389 non-null  int64    x   Drop column -- irrelevant for Single family homes which typically don't have pet restrictions in California 
 37  homeType            35389 non-null  object   x   Retain records where 'homeType'='SINGLE_FAMILY' and then drop this column 
 38  county              35389 non-null  object   x   Drop column -- in favor of countyId
 </pre>

## Data cleanup:
After conducting the data cleanup as identified in the Data Analysis section, we have a total 19,130 records with 15 columns:
<pre>
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   event              19130 non-null  object 
 1   price              19130 non-null  float64
 2   yearBuilt          19130 non-null  object 
 3   zipcode            19130 non-null  object 
 4   description        19130 non-null  object 
 5   livingAreaValue    19130 non-null  float64
 6   bathrooms          19130 non-null  float64
 7   bedrooms           19130 non-null  float64
 8   parking            19130 non-null  bool   
 9   garageSpaces       19130 non-null  float64
 10  levels             19130 non-null  object 
 11  pool               19130 non-null  bool   
 12  spa                19130 non-null  bool   
 13  isNewConstruction  19130 non-null  bool   
 14  month              19130 non-null  object 
dtypes: bool(4), float64(5), object(6)
memory usage: 1.7+ MB
</pre>

## Feature Engineering
1. **'description' field was mined using NLTK** to look for specific keywords (for Features like 'hardwood floors', 'solar panels' or for Property Styles like 'victorian', 'contemporary' or for Property Conditions like 'renovated') that might help identify features of single family homes that are not captured in the individual columns. Keywords searched were:
   - **"Features"**: ['hardwood floors', 'master suite', 'detached garage', 'waterfront', 'open floor plan', 'bonus room', 'rec room', 'game room', 'loft', 'sunroom', 'solar panels', 'office space', 'guest house', 'in-law suite', 'granite', 'high ceilings', 'updated electrical', 'new roof', 'new HVAC', 'custom landscaping']
   - **"Property_Condition"**: ['remodeled', 'renovated', 'designer', 'fixer-upper', 'needs TLC', 'handyman special', 'model home']
   - **"Location_Neighborhood"**: ['safe', 'cul-de-sac', 'gated', 'public transport', 'freeway', 'golf course', 'beach', 'great schools',  'excellent schools', 'family-friendly']
   - **"Property_Style_Layout"**: ['ranch', 'colonial', 'victorian', 'mediterranean', 'contemporary', 'modern', 'traditional', 'farmhouse']
   - **"Amenities"**: ['pet-friendly', 'furnished', 'community pool', 'gym', 'fireplace', 'tennis', 'basketball']
   - **"Terms"**: ['short sale', 'foreclosure', 'owner financing', 'seller financing', 'cash offer', 'deed restrictions']
   These keywords were added as columns to the DataFrame after stemming and lemmatizing and each occurence of a keyword in the 'description' field (tokenized after converting to lowercase and removing punctuations) incremented the value in that column for that row.

2. **Derived Features from existing data**:
   1. Age of the House: Calculate the age of the house by subtracting the year built from the reference year(2021).
   2. Number of Rooms: Combine the number of bedrooms and bathrooms to get a total room count.
   3. Living Area per Room: Divide the living area by the number of rooms to get a measure of space per room.
   4. Bath to Bedroom Ratio: bathrooms / bedrooms
   5. Garage to Bedroom Ratio: garageSpaces / bedrooms

3. **Drop columns** that could likely to increase collinearity. These were:
   1. yearBuilt - used in age of the house
   2. livingAreaValue - used in living area per room

4. **Encode categorical columns using One Hot Encoding**

## Models and their resulting metrics
GridSearchCV was used to perform hyperparameter tuning across:
1. Linear Regression with polynomail factors up to a degree=5
2. Ridge Regression
3. Lasso Regression
4. Gradient Descent

### Models comparison


| Model             | MSE                | R-squared |
|-------------------|--------------------|-----------|
| Linear Regression | 6.59442825355e+35  | -1.29     |
| Ridge Regression  | 2593400447890      | 0.49      |
| Lasso Regression  | 2593931454823      | 0.49      |
| Gradient Boosting | 1675809701616      | 0.67      |


- **Linear Regression:**
The negative R-squared values and the large MSEs for Linear Regression are a strong indication that there might be high collinearity in the dataset. 

- **Ridge Regression:**
The R-squared is 0.49. Not a strong R-squared metric. Also the MSE is quite high (2593400447890). So this model may not be the best fit.

- **Lasso Regression:**
The R-squared is also similar to Ridge Regression: 0.49. Again, not a strong R-squared metric. Also the MSE is quite high (2593931454823). So this model may not be the best fit either.

- **Gradient Boosting:**
The Gradient Boosting model provided these metrics:
  - **Mean Squared Error:** 1675809701616.251
  - **R-squared:** 0.67 <p>

- **Analysis in favor of Gradient Boosting:**
  The R-squared is much better than the Ridge or Lasso Regressions and also the MSE is relatively much smaller (1675809701616 vs. 2593400447890).<p>
  Hyper parameter tuning for Gradient Boosting faired a slightly better result compared to default model.

### Gradient Boosting is the best predictor model (in terms of MSE and R-squared).

## Feature Importance Analysis: Gradient Boosting Results
I have trained a Gradient Boosting model to predict house prices using a variety of features, including some derived features and one-hot encoded zip code data. The results are quite revealing:

#### Key Findings:
1. **Bathrooms Reign Supreme**: The most important feature by far is the number of bathrooms. This reinforces the well-known fact that more bathrooms are strongly associated with higher house prices.
2. **Space is Key**: The feature LivingAreaPerRoom, representing living area per room, has a significant importance score. It suggests that buyers value houses with ample space per room.
3. **Other Notable Factors**: NumRooms (total rooms) has a smaller but still noticeable importance. Interestingly, AgeOfHouse also has a relatively high importance, suggesting that older houses might be less desirable.
4. **Location is Crucial**: We see some zip codes with significant importance scores, such as zipcode_94301.0, zipcode_94027.0, and zipcode_92118.0. These zip codes likely represent areas with high demand, desirable amenities, or other factors driving up prices.
5. **Gold nuggets in the 'description' field**: Its interesting to note that proximity to beach and architectural styles (like the Victorian) play a significant factor too.

#### Interpreting the Results:
- The high importance scores for bathrooms, LivingAreaPerRoom, and Rooms are intuitive and align with common real estate knowledge.
- The significant influence of some zip codes, beaches and architectural styles highlights the importance of location, style and neighborhood factors.

#### Here are the top features with their importance scores:
<pre>
            feature  importance
         bathrooms    0.530823
 LivingAreaPerRoom    0.190601
          NumRooms    0.041782
          bedrooms    0.027198
     zipcode_94027    0.024922
        AgeOfHouse    0.023323
             beach    0.014779
     zipcode_94301    0.013488
            design    0.010695
           parking    0.009503
      garageSpaces    0.008584
         victorian    0.008423
     zipcode_93108    0.008220
BathToBedroomRatio    0.006726
     zipcode_90402    0.005847
</pre>

## Next Steps:
For improving the model further, a few of the below steps can be taken:
- **Apply further sophisticated models** like Neural Networks or perhaps other modeling techniques like Random Forest could also be considered. (Both of these technqiues have yet to be introduced in the course at the time of this writing)
- **Deep Dive on Key Features**: We should delve deeper into the factors that make specific zip codes more desirable. This could involve analyzing demographic data, crime rates, school quality, proximity to amenities, and other relevant factors.
- **Domain Expertise**: Consider seeking input from real estate experts to validate our findings and understand the local market dynamics.
- **Model Refinement**: We can continue to refine our model by incorporating other potentially influential features, like the Distance to Amenities, Property Tax Rate, or any other relevant factors. But that would require access to other datasets that are readily available and relatable to the current dataset as of this writing.

Overall, the feature importance analysis provides valuable insights into the factors driving house prices. By focusing on these key features and exploring the nuances of location, we can enhance our ability to predict house prices accurately.

It's important to remember that these findings are based on the data and model used, and they might vary depending on the dataset and the model's parameters.

## Key learnings applied in the Capstone project include:
- Data analysis
- Data cleansing
- Feature engineering
  - Natural Language Processing (NLTK, Tokenize, Stemming, Lemmatizing)
  - Feature derivations
  - Categorical value engineering using ColumnTransformers like OneHotEncoding
- Linear Regression with multiple polynomial factors, apply standard scaling
- Ridge Regression
- Lasso Regression
- Gradient Descent
- Compare models using metrics
- Apply hyperparameter tuning using GridSearchCV
- Feature importance derivation and definition
- Plot views using matplotlib and seaborn libraries
- Able to interpret findings and document/showcase findings in business terms