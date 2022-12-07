import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import statsmodels.api as sm
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from scipy.stats import zscore
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, plot_roc_curve, precision_recall_curve, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('PCOS_data.csv')
#df = pd.read_csv('/Users/almgacis/Documents/MSU/CMSE_830/HW/data/PCOS_data.csv')

#Replacing nans with median
df['Marriage Status (Yrs)'] = df['Marriage Status (Yrs)'].fillna(df['Marriage Status (Yrs)'].median())
df['II    beta-HCG(mIU/mL)'] = df['II    beta-HCG(mIU/mL)'].fillna(df['II    beta-HCG(mIU/mL)'].median())
df['AMH(ng/mL)'] = df['AMH(ng/mL)'].fillna(df['AMH(ng/mL)'].median())
df['Fast food (Y/N)'] = df['Fast food (Y/N)'].fillna(df['Fast food (Y/N)'].median())

#Dropping na, if any more
df2 = df.dropna()

#Averaging almost similar features
df2['Avg Follicle No'] = round((df2['Follicle No. (L)'] + df2['Follicle No. (R)'])/2)
df2['Avg F size (mm)'] = round((df2['Avg. F size (L) (mm)'] + df2['Avg. F size (R) (mm)'])/2)


st.sidebar.title("About the Data")
st.sidebar.markdown("Polycystic ovary syndrome (PCOS) is a condition involving irregular, missed, or prolonged periods, and most of the time, excess androgen levels. The ovaries develop follicles (small collections of fluid), and may fail to release eggs on a regular basis. The dataset contains physical and clinical variables that might help with determining PCOS diagnosis and infertility related issues. The data was collected from 10 hospitals across Kerala, India.")
st.sidebar.markdown('*The dataset entitled Polycystic ovary syndrome (PCOS) was made by Prasoon Kottarathil in 2020 and was published in [Kaggle](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos).*')

#-----Scaling / Removing Outliers--------------------------------------------------------------------------------------

st.markdown(""" <style> .font {
font-size:48px ; font-family: 'Cooper Black'; color: #9C6DA5;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Polycystic Ovarian Syndrome (PCOS) Diagnosis Data</p>', unsafe_allow_html=True)

# st.title("""# Polycystic Ovarian Syndrome (PCOS) Diagnosis Data""")

#eda = '<p style="font-family:Courier; color:Blue; font-size: 20px;">Exploratory Data Analysis/p>'
#st.markdown(original_title, unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Dimension Reduction", "Classification"])

with tab1:

    measurements = df2.drop(labels=["PCOS (Y/N)"], axis=1).columns.tolist()

    #Removing redundant features
    df3 = df2.drop(['Sl. No', 'Patient File No.', 'Hip(inch)', 'Waist(inch)', 'BMI',
                   'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
                   'Avg. F size (R) (mm)', 'FSH/LH', 'II    beta-HCG(mIU/mL)'], axis=1)

    df3_corr = df2.drop(['Sl. No', 'Patient File No.', 'Hip(inch)', 'Waist(inch)', 'BMI',
                   'Follicle No. (L)', 'Follicle No. (R)', 'Avg. F size (L) (mm)',
                   'Avg. F size (R) (mm)', 'FSH/LH', 'II    beta-HCG(mIU/mL)',], axis=1)
                   # 'Weight (Kg)', 'Marriage Status (Yrs)', 'Fast food (Y/N)'], axis=1)

    df3copy = df3.copy()
    categorical = df3copy.drop(labels=[' Age (yrs)',
     'Weight (Kg)',
     'Height(Cm) ',
     'Pulse rate(bpm) ',
     'RR (breaths/min)',
     'Hb(g/dl)',
     'Cycle length(days)',
     'Marriage Status (Yrs)',
     'No. of abortions',
     '  I   beta-HCG(mIU/mL)',
     'FSH(mIU/mL)',
     'LH(mIU/mL)',
     'Waist:Hip Ratio',
     'TSH (mIU/L)',
     'AMH(ng/mL)',
     'PRL(ng/mL)',
     'Vit D3 (ng/mL)',
     'PRG(ng/mL)',
     'RBS(mg/dl)',
     'BP _Systolic (mmHg)',
     'BP _Diastolic (mmHg)',
     'Endometrium (mm)',
     'Avg Follicle No', 'Avg F size (mm)'], axis=1).columns.tolist()

    for i in categorical:
        df3copy[i].replace([1, 0], ['Yes', 'No'], inplace=True)
        df3copy[i].replace([2, 4, 5], ['Regular', 'Irregular', 'Irregular'], inplace=True)
        df3copy[i].replace([11, 12, 13, 14, 15, 16, 17, 18], ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB', 'AB-'], inplace=True)

    cat_for_corr = df3.drop(labels=[' Age (yrs)',
     'Weight (Kg)',
     'Height(Cm) ',
     'Pulse rate(bpm) ',
     'RR (breaths/min)',
     'Hb(g/dl)',
     'Cycle length(days)',
     'Marriage Status (Yrs)',
     'No. of abortions',
     '  I   beta-HCG(mIU/mL)',
     'FSH(mIU/mL)',
     'LH(mIU/mL)',
     'Waist:Hip Ratio',
     'TSH (mIU/L)',
     'AMH(ng/mL)',
     'PRL(ng/mL)',
     'Vit D3 (ng/mL)',
     'PRG(ng/mL)',
     'RBS(mg/dl)',
     'BP _Systolic (mmHg)',
     'BP _Diastolic (mmHg)',
     'Endometrium (mm)',
     'Avg Follicle No', 'Avg F size (mm)'], axis=1).columns.tolist()

    numerical = df3.drop(labels=['PCOS (Y/N)',
     'Blood Group',
     'Pregnant(Y/N)',
     'Cycle(R/I)',
     'Weight gain(Y/N)',
     'hair growth(Y/N)',
     'Skin darkening (Y/N)',
     'Hair loss(Y/N)',
     'Pimples(Y/N)',
     'Fast food (Y/N)',
     'Reg.Exercise(Y/N)'], axis=1).columns.tolist()

    #Using Z-scaler
    df3z_nom = pd.concat([df3copy[categorical], df3[numerical].apply(zscore)], axis=1)
    df3z_cor = pd.concat([df3[cat_for_corr], df3[numerical].apply(zscore)], axis=1)

    #Using MinMax-scaler
    minmaxscale = MinMaxScaler()
    df3_2 = df3.copy()
    df3_2[numerical] = minmaxscale.fit_transform(df3_2[numerical])
    df3mm_nom = pd.concat([df3copy[categorical], df3_2[numerical]], axis=1)
    df3mm_cor = pd.concat([df3[cat_for_corr], df3_2[numerical]], axis=1)

    #Raw with removed outliers (for plots and corr)
    np.random.seed(33454)

    Q1 = df3[numerical].quantile(0.25)
    Q3 = df3[numerical].quantile(0.75)
    IQR = Q3 - Q1
    LB = Q1 - 1.5 * IQR
    UB = Q3 + 1.5 * IQR

    out = pd.DataFrame(df3[numerical][((df3[numerical] < LB) | (df3[numerical] > UB)).any(axis=1)])
    df3_outcon_nom = pd.concat([df3copy[categorical], df3[numerical], out], axis=1)
    df3_outcon_cor = pd.concat([df3[cat_for_corr], df3[numerical], out], axis=1)

    df3_wout_nom = df3_outcon_nom[df3_outcon_nom.isnull().any(1)].dropna(axis=1)
    df3_wout_cor = df3_outcon_cor[df3_outcon_cor.isnull().any(1)].dropna(axis=1)


    #Raw with removed outliers then scaled (for plots and corr)
    df3z_wout_nom = pd.concat([df3copy[categorical], df3_wout_nom[numerical].apply(zscore)], axis=1).drop(['No. of abortions'], axis=1).dropna()
    df3z_wout_cor = pd.concat([df3[cat_for_corr], df3_wout_cor[numerical].apply(zscore)], axis=1).drop(['No. of abortions'], axis=1).dropna()
 
    minmaxscale = MinMaxScaler()
    df3_wout_nom2 = df3_wout_nom.copy()
    df3_wout_cor2 = df3_wout_cor.copy()
    df3_wout_nom2[numerical] = minmaxscale.fit_transform(df3_wout_nom2[numerical])
    df3_wout_cor2[numerical] = minmaxscale.fit_transform(df3_wout_cor2[numerical])

    df3mm_wout_nom = pd.concat([df3copy[categorical], df3_wout_nom2[numerical]], axis=1).dropna()
    df3mm_wout_cor = pd.concat([df3[cat_for_corr], df3_wout_cor2[numerical]], axis=1).dropna()


    trans = st.sidebar.multiselect("Transform data: One scaler at a time", ["MinMax Scale", "Z-Score Scale", "Remove outliers"], default=["MinMax Scale"])
    if not trans:
        df4 = df3copy
        df4_corr = df3_corr
    elif "MinMax Scale" in trans:
        if "Remove outliers" in trans:
            df4 = df3mm_wout_nom
            df4_corr = df3mm_wout_cor
        elif "Z-Score Scale" in trans:
            st.sidebar.error("Please choose only one type of scaler at a time.")
            st.stop()
        else:
            df4 = df3mm_nom
            df4_corr = df3mm_cor
    elif "Z-Score Scale" in trans:
        if "Remove outliers" in trans:
            df4 = df3z_wout_nom
            df4_corr = df3z_wout_cor
        elif "MinMax Scale" in trans:
            st.error("Please choose only one type of scaler at a time.")
            st.stop()
        else:
            df4 = df3z_nom
            df4_corr = df3z_cor
    elif "Remove outliers" in trans:
        if "Z-Score Scale" in trans:
            df4 = df3z_wout_nom
            df4_corr = df3z_wout_cor
        elif "MinMax Scale" in trans:
            df4 = df3mm_wout_nom
            df4_corr = df3mm_wout_cor
        else:
            df4 = df3_wout_nom
            df4_corr = df3_wout_cor


    #---Summary Plots---------------------------------------------------------------------------------------

    def filter_dataframe(df4: pd.DataFrame) -> pd.DataFrame:
            modify = st.sidebar.checkbox("Add filters for univariate and bivariate plots")
            if not modify:
                return df4

            df4 = df4.copy()

            modification_container = st.container()

            with modification_container:
                to_filter_columns = st.sidebar.multiselect("Filter dataframe on", df4.columns)
                for column in to_filter_columns:
                    left, right = st.sidebar.columns((1, 20))
                    left.write("↳")
                    if is_categorical_dtype(df4[column]) or df4[column].nunique() < 10:
                        user_cat_input = right.multiselect(
                            f"Values for {column}",
                            df4[column].unique(),
                            default=list(df4[column].unique()),
                        )
                        df4 = df4[df4[column].isin(user_cat_input)]
                    elif is_numeric_dtype(df4[column]):
                        _min = float(df4[column].min())
                        _max = float(df4[column].max())
                        step = (_max - _min) / 100
                        user_num_input = right.slider(
                            f"Values for {column}",
                            min_value=_min,
                            max_value=_max,
                            value=(_min, _max),
                            step=step,
                        )
                        df4 = df4[df4[column].between(*user_num_input)]
                    else:
                        user_text_input = right.text_input(
                            f"Substring or regex in {column}",
                        )
                        if user_text_input:
                            df4 = df4[df4[column].astype(str).str.contains(user_text_input)]
            return df4

    df5 = filter_dataframe(df4)

    data_show1 = st.checkbox('Show/hide data head')
    if data_show1:
        st.dataframe(df5.head(5))

    data_show2 = st.checkbox('Show/hide summary statistics of numerical features')
    if data_show2:
        st.dataframe(df5.describe())

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    col1, col2 = st.columns(2,gap='large')

    with col1:
        st.subheader("""Bar Plot of categorical features by diagnosis""")
        cat_x1 = st.selectbox('Select symptom',categorical)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        fig1 = px.histogram(df5, x="PCOS (Y/N)", color=cat_x1, barmode='group')
        fig1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },font=dict(
            # family="Courier New, monospace",
            size=12,
            # color="RebeccaPurple"
            ),autosize=False,
            width=350,
            height=450,
        )

        if cat_x1 ==   'PCOS (Y/N)'    :
            st.caption('This indicates if the patient has PCOS or not.')        
        if cat_x1 ==    'Blood Group'   :
            st.caption("One study claims that females with a blood type O positive have the highest chance of developing PCOS, followed by blood type B positive, while Rh negative didn’t have any relationship with PCOS.")       
        if cat_x1 ==    'Weight gain(Y/N)'  :
            st.caption("Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.")        
        if cat_x1 ==    'Pregnant(Y/N)' :
            st.caption("PCOS generates higher risks for women during pregnancy. Not only does it affect the mother, it also affects the child. Women with PCOS are more likely to miscarry in the first few months of pregnancy than those without.")       
        if cat_x1 ==    'hair growth(Y/N)'  :
            st.caption("One symptom of PCOS is hirsutism, which is excess facial and body hair.")       
        if cat_x1 ==    'Skin darkening (Y/N)'  :
            st.caption("PCOS can cause your skin to have dark patches. This is due to the insulin resistance experienced by those with PCOS.")      
        if cat_x1 ==    'Hair loss(Y/N)'    :
            st.caption("Some women with PCOS experience hair thinning and hair loss.")      
        if cat_x1 ==    'Pimples(Y/N)'  :
            st.caption("PCOS causes the ovaries to produce more androgens, which triggers the production of oil in the skin, which leads to acne.")     
        if cat_x1 ==    'Fast food (Y/N)'   :
            st.caption("Women with PCOS are advised to avoid saturated and trans fats, refined carbohydrates, sugar, dairy, and alcohol. A healthier diet can help manage the PCOS symptoms and reach or manage a healthy weight.")     
        if cat_x1 ==    'Reg.Exercise(Y/N)' :
            st.caption("Exercising has a lot of benefits for those who have PCOS. Reducing insulin resistance, stabilizing mood, improving fertility, and weight loss are just some of the benefits that can be gained through exercise.")            
        if cat_x1 == 'Cycle(R/I)':
            st.caption('Many women with PCOS experience irregular periods. But some women with PCOS have regular periods.')
        
        st.write(fig1)

    with col2:
        st.subheader("""Violin Plot of numerical features by diagnosis""")
        num_y1 = st.selectbox('Compare diagnosis for what feature?', numerical)
        st.set_option('deprecation.showPyplotGlobalUse', False)

        if num_y1:
            fig2 = px.violin(df5, x="PCOS (Y/N)", y=num_y1, box=True, hover_data=df5.columns)
            fig2.update_layout({
            'plot_bgcolor': 'rgba(0, 0, 0, 0)',
            'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            },font=dict(
                size=12
                ),autosize=False,
                width=450,
                height=450,
            )

        if num_y1 == ' Age (yrs)':
            st.caption('PCOS can be detected any time after puberty. Majority of women find out that they have PCOS when they are in their 20s and 30s.')
        if num_y1 == 'Weight (Kg)':
            st.caption('Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.')
        if num_y1 == 'Height(Cm) ':
            st.caption('Height can be a factor when it comes to PCOS. A study has shown that height is positively related with PCOS. Girls that are taller have a higher chance of developing PCOS than girls of average height.')
        if num_y1 == 'Pulse rate(bpm) ':
            st.caption('The range that is considered as normal for adults is 60 to 100 beats per minute. Having a lower heart rate means that your heart is more efficient and you have better cardiovascular fitness.')
        if num_y1 == 'RR (breaths/min)':
            st.caption('The range that is considered as normal for adults is 12-16 breaths per minute. A study states that women with PCOS are about 10% more likely to have lower lung function.')
        if num_y1 == 'Hb(g/dl)':
            st.caption('This is the amount of hemoglobin in your blood. The normal range for female adults is 12.1 to 15.1 g/dL. It is reported that women with PCOS have hemoglobin levels that are significantly higher.')
        if num_y1 == 'Cycle length(days)':
            st.caption('On average, the menstrual cycle is 21-35 days. Irregular periods are those that last below 21 days or longer than 35 days.')
        if num_y1 == 'Marriage Status (Yrs)':
            st.caption('PCOS can affect your relationship with your partner. Intimacy and pregnancy for someone with PCOS can be difficult and it can be reasons why some relationships struggle.')
        if num_y1 == 'No. of abortions':
            st.caption('Those who have PCOS have a high abortion rate of 30-50% in the first trimester, chances of recurrent early abortion is 36-82%, and habitual abortion is 58%.')
        if num_y1 == '  I   beta-HCG(mIU/mL)':
            st.caption('This is a hormone that the body produces during pregnancy. For adult women, a level of less than 5 mIU/mL is considered normal and it means that you are unlikely to be pregnant. Having a result of more than 20 mIU/mL means that you are likely to be pregnant.')
        if num_y1 == 'FSH(mIU/mL)':
            st.caption('This is a hormone that helps control the menstrual cycle. It also stimulates the growth of eggs in the ovaries. The normal levels for female are before puberty: 0 to 4.0 mIU/mL, during puberty: 0.3 to 10.0 mIU/mL, women who are still menstruating: 4.7 to 21.5 mIU/mL, and after menopause: 25.8 to 134.8 mIU/mL. High levels of FSH might suggest that you have PCOS.')
        if num_y1 == 'LH(mIU/mL)':
            st.caption('This is a hormone that the pituitary releases during ovulation. The normal levels for women depends on the phase of the menstrual cycle. The levels are follicular phase of menstrual cycle: 1.68 to 15 IU/mL, midcycle peak: 21.9 to 56.6 IU/mL, luteal phase: 0.61 to 16.3 IU/mL, postmenopausal: 14.2 to 52.3 IU/mL. Many women with PCOS have LH within the range of 5-20 mIU/mL.')
        if num_y1 == 'Waist:Hip Ratio':
            st.caption('Waist to hip ratio correspond to a high chance PCOS.')
        if num_y1 == 'TSH (mIU/L)':
            st.caption('This is a hormone that helps control the production of hormones and the metabolism of your body. The normal levels are from 0.4 to 4.0 mIU/L if you have no symptoms of having an under- or over-active thyroid. If you are receiving treatment for a thyroid disorder, TSH levels should be between 0.5 and 2.0 mIU/L. Women with PCOS generally have normal TSH levels.')
        if num_y1 == 'AMH(ng/mL)':
            st.caption('This is a hormone that is used to measure a woman’s ovarian reserve. The normal levels depends on the age of the woman -- under 33 years old: 2.1 – 6.8 ng/ml, 33 - 37 years old: 1.7 – 3.5 ng/ml, 38 - 40 years old: 1.1 – 3.0 ng/ml, 41+ years old: 0.5 – 2.5 ng/ml. An AMH above 6.8 ng/ml is considered high and is a potential sign of PCOS at any age.')
        if num_y1 == 'PRL(ng/mL)':
            st.caption('This is a hormone that triggers breast development and breast milk production in women. The normal range for non-pregnant women is less than 25 ng/mL , while it is 80 to 400 ng/mL for pregnant women. Women with PCOS usually have normal prolactin levels.')
        if num_y1 == 'Vit D3 (ng/mL)':
            st.caption('Vitamin D values are defined as normal : ≥20 ng/mL, vitamin D insufficiency: 12 to 20 ng/mL, and vitamin D deficiency: less than 12 ng/mL. Women with PCOS have a relatively high incidence of vitamin D deficiency. Vitamin D deficiency could aggravate some PCOS symptoms.')
        if num_y1 == 'PRG(ng/mL)':
            st.caption('Progesterone aids the uterus in getting ready so it can host a fertilized egg during pregnancy. The normal ranges are prepubescent girls: 0.1 to 0.3 ng/mL, follicular stage of the menstrual cycle: 0.1 to 0.7 ng/mL, luteal stage of the menstrual cycle: 2 to 25 ng/mL, first trimester of pregnancy: 10 to 44 ng/mL, second trimester of pregnancy: 19.5 to 82.5 ng/mL, third trimester of pregnancy: 65 to 290 ng/mL')
        if num_y1 == 'RBS(mg/dl)':
            st.caption('Blood sugar levels are defined normal: less than 140 mg/dL, prediabetes: between 140 and 199 mg/dL, diabetes: more than 200 mg/dL. In a random blood sugar test, a result of 200 mg/dL or higher would indicate diabetes. More than 50% of women with PCOS develop type 2 diabetes by 40 years old.')
        if num_y1 == 'BP _Systolic (mmHg)':
            st.caption('This is the maximum pressure your heart exerts while it is beating. A systolic pressure that is above 90 mm Hg and less than 120 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
        if num_y1 == 'BP _Diastolic (mmHg)':
            st.caption('This is the amount of pressure in the arteries between beats. A diastolic pressure that is above 60 mm Hg and less than 80 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
        if num_y1 == 'Endometrium (mm)':
            st.caption('Endometrial thickness of more than 8.5 mm could be linked with endometrial disease in women with PCOS.')
        if num_y1 == 'Avg Follicle No':
            st.caption('Having an antral follicle count of 6-10 means that a woman has a normal ovarian reserve. A count of less than 6 is considered low, while a count of greater than 12 is considered high. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')
        if num_y1 == 'Avg F size (mm)':
            st.caption('A regular ovary comprises of 8-10 follicles ranging from 2 to 28 mm. Follicles sized less than 18 mm are called antral follicles, while follicles sized 18 to 28 mm are called dominant follicles. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')

        st.write(fig2)

    #---Bivariate---------------------------------------------------------------------------------------

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    l1, m1, r1 = st.columns((2,5,1))
    with m1:
        st.subheader("Bivariate scatterplot by diagnosis")

    col3, col4, col5 = st.columns(3,gap='large')

    with col3:
        alt_x = st.selectbox("Compare diagnosis for which features (X)?", numerical)
        if alt_x == ' Age (yrs)':
            st.caption('PCOS can be detected any time after puberty. Majority of women find out that they have PCOS when they are in their 20s and 30s.')
        if alt_x == 'Weight (Kg)':
            st.caption('Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.')
        if alt_x == 'Height(Cm) ':
            st.caption('Height can be a factor when it comes to PCOS. A study has shown that height is positively related with PCOS. Girls that are taller have a higher chance of developing PCOS than girls of average height.')
        if alt_x == 'Pulse rate(bpm) ':
            st.caption('The range that is considered as normal for adults is 60 to 100 beats per minute. Having a lower heart rate means that your heart is more efficient and you have better cardiovascular fitness.')
        if alt_x == 'RR (breaths/min)':
            st.caption('The range that is considered as normal for adults is 12-16 breaths per minute. A study states that women with PCOS are about 10% more likely to have lower lung function.')
        if alt_x == 'Hb(g/dl)':
            st.caption('This is the amount of hemoglobin in your blood. The normal range for female adults is 12.1 to 15.1 g/dL. It is reported that women with PCOS have hemoglobin levels that are significantly higher.')
        if alt_x == 'Cycle length(days)':
            st.caption('On average, the menstrual cycle is 21-35 days. Irregular periods are those that last below 21 days or longer than 35 days.')
        if alt_x == 'Marriage Status (Yrs)':
            st.caption('PCOS can affect your relationship with your partner. Intimacy and pregnancy for someone with PCOS can be difficult and it can be reasons why some relationships struggle.')
        if alt_x == 'No. of abortions':
            st.caption('Those who have PCOS have a high abortion rate of 30-50% in the first trimester, chances of recurrent early abortion is 36-82%, and habitual abortion is 58%.')
        if alt_x == '  I   beta-HCG(mIU/mL)':
            st.caption('This is a hormone that the body produces during pregnancy. For adult women, a level of less than 5 mIU/mL is considered normal and it means that you are unlikely to be pregnant. Having a result of more than 20 mIU/mL means that you are likely to be pregnant.')
        if alt_x == 'FSH(mIU/mL)':
            st.caption('This is a hormone that helps control the menstrual cycle. It also stimulates the growth of eggs in the ovaries. The normal levels for female are before puberty: 0 to 4.0 mIU/mL, during puberty: 0.3 to 10.0 mIU/mL, women who are still menstruating: 4.7 to 21.5 mIU/mL, and after menopause: 25.8 to 134.8 mIU/mL. High levels of FSH might suggest that you have PCOS.')
        if alt_x == 'LH(mIU/mL)':
            st.caption('This is a hormone that the pituitary releases during ovulation. The normal levels for women depends on the phase of the menstrual cycle. The levels are follicular phase of menstrual cycle: 1.68 to 15 IU/mL, midcycle peak: 21.9 to 56.6 IU/mL, luteal phase: 0.61 to 16.3 IU/mL, postmenopausal: 14.2 to 52.3 IU/mL. Many women with PCOS have LH within the range of 5-20 mIU/mL.')
        if alt_x == 'Waist:Hip Ratio':
            st.caption('Waist to hip ratio correspond to a high chance PCOS.')
        if alt_x == 'TSH (mIU/L)':
            st.caption('This is a hormone that helps control the production of hormones and the metabolism of your body. The normal levels are from 0.4 to 4.0 mIU/L if you have no symptoms of having an under- or over-active thyroid. If you are receiving treatment for a thyroid disorder, TSH levels should be between 0.5 and 2.0 mIU/L. Women with PCOS generally have normal TSH levels.')
        if alt_x == 'AMH(ng/mL)':
            st.caption('This is a hormone that is used to measure a woman’s ovarian reserve. The normal levels depends on the age of the woman -- under 33 years old: 2.1 – 6.8 ng/ml, 33 - 37 years old: 1.7 – 3.5 ng/ml, 38 - 40 years old: 1.1 – 3.0 ng/ml, 41+ years old: 0.5 – 2.5 ng/ml. An AMH above 6.8 ng/ml is considered high and is a potential sign of PCOS at any age.')
        if alt_x == 'PRL(ng/mL)':
            st.caption('This is a hormone that triggers breast development and breast milk production in women. The normal range for non-pregnant women is less than 25 ng/mL , while it is 80 to 400 ng/mL for pregnant women. Women with PCOS usually have normal prolactin levels.')
        if alt_x == 'Vit D3 (ng/mL)':
            st.caption('Vitamin D values are defined as normal : ≥20 ng/mL, vitamin D insufficiency: 12 to 20 ng/mL, and vitamin D deficiency: less than 12 ng/mL. Women with PCOS have a relatively high incidence of vitamin D deficiency. Vitamin D deficiency could aggravate some PCOS symptoms.')
        if alt_x == 'PRG(ng/mL)':
            st.caption('Progesterone aids the uterus in getting ready so it can host a fertilized egg during pregnancy. The normal ranges are prepubescent girls: 0.1 to 0.3 ng/mL, follicular stage of the menstrual cycle: 0.1 to 0.7 ng/mL, luteal stage of the menstrual cycle: 2 to 25 ng/mL, first trimester of pregnancy: 10 to 44 ng/mL, second trimester of pregnancy: 19.5 to 82.5 ng/mL, third trimester of pregnancy: 65 to 290 ng/mL')
        if alt_x == 'RBS(mg/dl)':
            st.caption('Blood sugar levels are defined normal: less than 140 mg/dL, prediabetes: between 140 and 199 mg/dL, diabetes: more than 200 mg/dL. In a random blood sugar test, a result of 200 mg/dL or higher would indicate diabetes. More than 50% of women with PCOS develop type 2 diabetes by 40 years old.')
        if alt_x == 'BP _Systolic (mmHg)':
            st.caption('This is the maximum pressure your heart exerts while it is beating. A systolic pressure that is above 90 mm Hg and less than 120 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
        if alt_x == 'BP _Diastolic (mmHg)':
            st.caption('This is the amount of pressure in the arteries between beats. A diastolic pressure that is above 60 mm Hg and less than 80 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
        if alt_x == 'Endometrium (mm)':
            st.caption('Endometrial thickness of more than 8.5 mm could be linked with endometrial disease in women with PCOS.')
        if alt_x == 'Avg Follicle No':
            st.caption('Having an antral follicle count of 6-10 means that a woman has a normal ovarian reserve. A count of less than 6 is considered low, while a count of greater than 12 is considered high. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')
        if alt_x == 'Avg F size (mm)':
            st.caption('A regular ovary comprises of 8-10 follicles ranging from 2 to 28 mm. Follicles sized less than 18 mm are called antral follicles, while follicles sized 18 to 28 mm are called dominant follicles. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')

    with col4:
        alt_y = st.selectbox("Compare diagnosis for which features? (Y)", numerical)
        if alt_y == ' Age (yrs)':
            st.caption('PCOS can be detected any time after puberty. Majority of women find out that they have PCOS when they are in their 20s and 30s.')
        if alt_y == 'Weight (Kg)':
            st.caption('Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.')
        if alt_y == 'Height(Cm) ':
            st.caption('Height can be a factor when it comes to PCOS. A study has shown that height is positively related with PCOS. Girls that are taller have a higher chance of developing PCOS than girls of average height.')
        if alt_y == 'Pulse rate(bpm) ':
            st.caption('The range that is considered as normal for adults is 60 to 100 beats per minute. Having a lower heart rate means that your heart is more efficient and you have better cardiovascular fitness.')
        if alt_y == 'RR (breaths/min)':
            st.caption('The range that is considered as normal for adults is 12-16 breaths per minute. A study states that women with PCOS are about 10% more likely to have lower lung function.')
        if alt_y == 'Hb(g/dl)':
            st.caption('This is the amount of hemoglobin in your blood. The normal range for female adults is 12.1 to 15.1 g/dL. It is reported that women with PCOS have hemoglobin levels that are significantly higher.')
        if alt_y == 'Cycle length(days)':
            st.caption('On average, the menstrual cycle is 21-35 days. Irregular periods are those that last below 21 days or longer than 35 days.')
        if alt_y == 'Marriage Status (Yrs)':
            st.caption('PCOS can affect your relationship with your partner. Intimacy and pregnancy for someone with PCOS can be difficult and it can be reasons why some relationships struggle.')
        if alt_y == 'No. of abortions':
            st.caption('Those who have PCOS have a high abortion rate of 30-50% in the first trimester, chances of recurrent early abortion is 36-82%, and habitual abortion is 58%.')
        if alt_y == '  I   beta-HCG(mIU/mL)':
            st.caption('This is a hormone that the body produces during pregnancy. For adult women, a level of less than 5 mIU/mL is considered normal and it means that you are unlikely to be pregnant. Having a result of more than 20 mIU/mL means that you are likely to be pregnant.')
        if alt_y == 'FSH(mIU/mL)':
            st.caption('This is a hormone that helps control the menstrual cycle. It also stimulates the growth of eggs in the ovaries. The normal levels for female are before puberty: 0 to 4.0 mIU/mL, during puberty: 0.3 to 10.0 mIU/mL, women who are still menstruating: 4.7 to 21.5 mIU/mL, and after menopause: 25.8 to 134.8 mIU/mL. High levels of FSH might suggest that you have PCOS.')
        if alt_y == 'LH(mIU/mL)':
            st.caption('This is a hormone that the pituitary releases during ovulation. The normal levels for women depends on the phase of the menstrual cycle. The levels are follicular phase of menstrual cycle: 1.68 to 15 IU/mL, midcycle peak: 21.9 to 56.6 IU/mL, luteal phase: 0.61 to 16.3 IU/mL, postmenopausal: 14.2 to 52.3 IU/mL. Many women with PCOS have LH within the range of 5-20 mIU/mL.')
        if alt_y == 'Waist:Hip Ratio':
            st.caption('Waist to hip ratio correspond to a high chance PCOS.')
        if alt_y == 'TSH (mIU/L)':
            st.caption('This is a hormone that helps control the production of hormones and the metabolism of your body. The normal levels are from 0.4 to 4.0 mIU/L if you have no symptoms of having an under- or over-active thyroid. If you are receiving treatment for a thyroid disorder, TSH levels should be between 0.5 and 2.0 mIU/L. Women with PCOS generally have normal TSH levels.')
        if alt_y == 'AMH(ng/mL)':
            st.caption('This is a hormone that is used to measure a woman’s ovarian reserve. The normal levels depends on the age of the woman -- under 33 years old: 2.1 – 6.8 ng/ml, 33 - 37 years old: 1.7 – 3.5 ng/ml, 38 - 40 years old: 1.1 – 3.0 ng/ml, 41+ years old: 0.5 – 2.5 ng/ml. An AMH above 6.8 ng/ml is considered high and is a potential sign of PCOS at any age.')
        if alt_y == 'PRL(ng/mL)':
            st.caption('This is a hormone that triggers breast development and breast milk production in women. The normal range for non-pregnant women is less than 25 ng/mL , while it is 80 to 400 ng/mL for pregnant women. Women with PCOS usually have normal prolactin levels.')
        if alt_y == 'Vit D3 (ng/mL)':
            st.caption('Vitamin D values are defined as normal : ≥20 ng/mL, vitamin D insufficiency: 12 to 20 ng/mL, and vitamin D deficiency: less than 12 ng/mL. Women with PCOS have a relatively high incidence of vitamin D deficiency. Vitamin D deficiency could aggravate some PCOS symptoms.')
        if alt_y == 'PRG(ng/mL)':
            st.caption('Progesterone aids the uterus in getting ready so it can host a fertilized egg during pregnancy. The normal ranges are prepubescent girls: 0.1 to 0.3 ng/mL, follicular stage of the menstrual cycle: 0.1 to 0.7 ng/mL, luteal stage of the menstrual cycle: 2 to 25 ng/mL, first trimester of pregnancy: 10 to 44 ng/mL, second trimester of pregnancy: 19.5 to 82.5 ng/mL, third trimester of pregnancy: 65 to 290 ng/mL')
        if alt_y == 'RBS(mg/dl)':
            st.caption('Blood sugar levels are defined normal: less than 140 mg/dL, prediabetes: between 140 and 199 mg/dL, diabetes: more than 200 mg/dL. In a random blood sugar test, a result of 200 mg/dL or higher would indicate diabetes. More than 50% of women with PCOS develop type 2 diabetes by 40 years old.')
        if alt_y == 'BP _Systolic (mmHg)':
            st.caption('This is the maximum pressure your heart exerts while it is beating. A systolic pressure that is above 90 mm Hg and less than 120 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
        if alt_y == 'BP _Diastolic (mmHg)':
            st.caption('This is the amount of pressure in the arteries between beats. A diastolic pressure that is above 60 mm Hg and less than 80 mm Hg is the normal range. Many symptoms linked with PCOS can cause blood pressure to increase.')
        if alt_y == 'Endometrium (mm)':
            st.caption('Endometrial thickness of more than 8.5 mm could be linked with endometrial disease in women with PCOS.')
        if alt_y == 'Avg Follicle No':
            st.caption('Having an antral follicle count of 6-10 means that a woman has a normal ovarian reserve. A count of less than 6 is considered low, while a count of greater than 12 is considered high. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')
        if alt_y == 'Avg F size (mm)':
            st.caption('A regular ovary comprises of 8-10 follicles ranging from 2 to 28 mm. Follicles sized less than 18 mm are called antral follicles, while follicles sized 18 to 28 mm are called dominant follicles. The threshold in determining PCOS is 12 or more follicles measuring 2-9 mm.')

    with col5:
        cat_hue = st.selectbox("Choose hue", categorical)
        if cat_hue ==   'PCOS (Y/N)'    :
            st.caption('This indicates if the patient has PCOS or not.')        
        if cat_hue ==   'Blood Group'   :
            st.caption("One study claims that females with a blood type O positive have the highest chance of developing PCOS, followed by blood type B positive, while Rh negative didn’t have any relationship with PCOS.")       
        if cat_hue ==   'Weight gain(Y/N)'  :
            st.caption("Weight gain and obesity are closely related with PCOS. 38%-88% of women that have PCOS are either overweight or obese.")        
        if cat_hue ==   'Pregnant(Y/N)' :
            st.caption("PCOS generates higher risks for women during pregnancy. Not only does it affect the mother, it also affects the child. Women with PCOS are more likely to miscarry in the first few months of pregnancy than those without.")       
        if cat_hue ==   'hair growth(Y/N)'  :
            st.caption("One symptom of PCOS is hirsutism, which is excess facial and body hair.")       
        if cat_hue ==   'Skin darkening (Y/N)'  :
            st.caption("PCOS can cause your skin to have dark patches. This is due to the insulin resistance experienced by those with PCOS.")      
        if cat_hue ==   'Hair loss(Y/N)'    :
            st.caption("Some women with PCOS experience hair thinning and hair loss.")      
        if cat_hue ==   'Pimples(Y/N)'  :
            st.caption("PCOS causes the ovaries to produce more androgens, which triggers the production of oil in the skin, which leads to acne.")     
        if cat_hue ==   'Fast food (Y/N)'   :
            st.caption("Women with PCOS are advised to avoid saturated and trans fats, refined carbohydrates, sugar, dairy, and alcohol. A healthier diet can help manage the PCOS symptoms and reach or manage a healthy weight.")     
        if cat_hue ==   'Reg.Exercise(Y/N)' :
            st.caption("Exercising has a lot of benefits for those who have PCOS. Reducing insulin resistance, stabilizing mood, improving fertility, and weight loss are just some of the benefits that can be gained through exercise.")      
        if cat_hue == 'Cycle(R/I)':
            st.caption('Many women with PCOS experience irregular periods. But some women with PCOS have regular periods.')

    if alt_x and alt_y and cat_hue:
        fig3 = px.scatter(df5, alt_x, alt_y, color=cat_hue, trendline="ols")
        fig3.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },font=dict(
            size=18
            )   
        )
    st.write(fig3)

    #---Correlation---------------------------------------------------------------------------------------

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    l3, m3, r3 = st.columns((4,5,1))
    with m3:
        st.subheader("Correlation")

    with st.form("key2"):

        corr_range = st.slider("Select correlation magnitude range", value=[0.0, 1.0], step=0.05)

        so = pd.DataFrame(df4_corr.corr().unstack().sort_values(kind="quicksort"), columns=['corrcoeff'])
        so.reset_index(inplace=True)
        soc = so.rename(columns = {'level_0':'Var1', 'level_1':'Var2'})
        socorr = soc[soc['Var1'] != soc['Var2']]
        selected_corr = socorr.where(abs(socorr['corrcoeff']) >= min(corr_range)).where(abs(socorr['corrcoeff']) <= max(corr_range)).dropna()
        filtered_corr_vars = selected_corr['Var1'].unique().tolist()
        selected_corr_data = df4_corr[filtered_corr_vars]

        button2 = st.form_submit_button("Apply range")

    corr_mat = st.checkbox('Show/hide correlation matrix')
    cor_pal = (sns.color_palette("colorblind", as_cmap=True))
    plt.tick_params(axis='both', which='major', labelsize=14)
    if corr_mat:
        fig4 = px.imshow(round(selected_corr_data.corr(),2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale=px.colors.sequential.Bluered)
        fig4.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },font=dict(
            size=12
            ),autosize=False,
                width=750,
                height=750, 
        )
        st.write(fig4)

    sources = st.sidebar.checkbox("Sources")
    links = ["[PCOS](https://www.hopkinsmedicine.org/health/conditions-and-diseases/polycystic-ovary-syndrome-pcos#:~:text=PCOS%20is%20a%20very%20common,%2C%20infertility%2C%20and%20weight%20gain.)",
                "[Age (yrs)](https://www.womenshealth.gov/a-z-topics/polycystic-ovary-syndrome#:~:text=However%2C%20their%20PCOS%20hormonal%20imbalance,with%20PCOS%20than%20those%20without.)",
                "[Weight (Kg)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6734597/#:~:text=In%20women%20who%20are%20genetically,are%20either%20overweight%20or%20obese.)",
                "[Height(Cm)](https://pubmed.ncbi.nlm.nih.gov/33979806/#:~:text=Girls%20who%20were%20persistently%20tall,girls%20with%20average%20height%20growth.)",
                "[Waist:Hip Ratio](https://journals.sagepub.com/doi/abs/10.1177/00369330211043206)",
                "[Marriage Status (Yrs)](https://www.verywellhealth.com/how-pcos-affects-your-relationships-2616703#:~:text=Infertility%2C%20or%20difficulty%20getting%20pregnant,comes%20with%20being%20a%20couple.)",
                "[Pulse rate(bpm)](https://www.mayoclinic.org/healthy-lifestyle/fitness/expert-answers/heart-rate/faq-20057979#:~:text=A%20normal%20resting%20heart%20rate%20for%20adults%20ranges%20from%2060,to%2040%20beats%20per%20minute.)",
                "[RR (breaths/min)1](https://www.hopkinsmedicine.org/health/conditions-and-diseases/vital-signs-body-temperature-pulse-rate-respiration-rate-blood-pressure#:~:text=Normal%20respiration%20rates%20for%20an,to%2016%20breaths%20per%20minute.)",
                "[RR (breaths/min)2](https://www.eurekalert.org/news-releases/521225#:~:text=Researchers%20used%20genetic%20variants%20associated,function%2C%20compared%20to%20other%20women)",
                "[BP _Systolic (mmHg)1](https://www.healthline.com/health/high-blood-pressure-hypertension/blood-pressure-reading-explained)",
                "[BP _Systolic (mmHg)2](https://www.ahajournals.org/doi/full/10.1161/hypertensionaha.107.088138#:~:text=Many%20of%20the%20symptoms%20associated,resistance%20and%20type%202%20diabetes.)",
                "[BP _Diastolic (mmHg)1](https://www.healthline.com/health/high-blood-pressure-hypertension/blood-pressure-reading-explained)",
                "[BP _Diastolic (mmHg)2](https://www.ahajournals.org/doi/full/10.1161/hypertensionaha.107.088138#:~:text=Many%20of%20the%20symptoms%20associated,resistance%20and%20type%202%20diabetes)",
                "[Blood Group](https://www.ijmrhs.com/medical-research/polycystic-ovary-syndrome-blood-group--diet-a-correlative-study-insouth-indian-females.pdf)",
                "[Hb(g/dl)1](https://www.ucsfhealth.org/medical-tests/hemoglobin#:~:text=Normal%20Results&text=Female%3A%2012.1%20to%2015.1%20g,121%20to%20151%20g%2FL)",
                "[Hb(g/dl)2](https://journals.sagepub.com/doi/10.1177/0300060520952282#:~:text=However%2C%20Han%20et%20al.10,dependent%20stimulatory%20effect%20on%20erythropoiesis)",
                "[I beta-HCG(mIU/mL)](https://www.urmc.rochester.edu/encyclopedia/content.aspx?contenttypeid=167&contentid=hcg_urine)",
                "[FSH(mIU/mL)1](https://medlineplus.gov/lab-tests/follicle-stimulating-hormone-fsh-levels-test/)",
                "[FSH(mIU/mL)2](https://www.mountsinai.org/health-library/tests/follicle-stimulating-hormone-fsh-blood-test)",
                "[LH(mIU/mL)1](https://www.urmc.rochester.edu/encyclopedia/content.aspx?ContentTypeID=167&ContentID=luteinizing_hormone_blood#:~:text=Here%20are%20normal%20ranges%3A,21.9%20to%2056.6%20IU%2FmL)",
                "[LH(mIU/mL)2](https://medlineplus.gov/lab-tests/luteinizing-hormone-lh-levels-test/#:~:text=LH%20plays%20an%20important%20role,This%20is%20known%20as%20ovulation.)",
                "[LH(mIU/mL)3](https://www.contemporaryobgyn.net/view/hormone-levels-and-pcos)",
                "[TSH (mIU/L)1](https://www.uclahealth.org/medical-services/surgery/endocrine-surgery/patient-resources/patient-education/endocrine-surgery-encyclopedia/tsh-test#:~:text=When%20a%20thyroid%20disorder%20is,0.5%20and%202.0%20mIU%2FL.)",
                "[TSH (mIU/L)2](https://www.yourhormones.info/hormones/thyroid-stimulating-hormone/)",
                "[TSH (mIU/L)3](https://www.contemporaryobgyn.net/view/hormone-levels-and-pcos)",
                "[AMH(ng/mL)1](https://rmanetwork.com/blog/anti-mullerian-hormone-amh-testing-of-ovarian-reserve/)",
                "[AMH(ng/mL)2](https://www.whitelotusclinic.ca/amh-pcos-test/)",
                "[AMH(ng/mL)3](https://www.cancer.gov/publications/dictionaries/cancer-terms/def/anti-mullerian-hormone)",
                "[PRL(ng/mL)1](https://www.mountsinai.org/health-library/tests/prolactin-blood-test#:~:text=Test%20is%20Performed-,Prolactin%20is%20a%20hormone%20released%20by%20the%20pituitary%20gland.,and%20milk%20production%20in%20women.)",
                "[PRL(ng/mL)2](https://www.ucsfhealth.org/medical-tests/prolactin-blood-test#:~:text=Normal%20Results&text=Men%3A%20less%20than%2020%20ng,80%20to%20400%20%C2%B5g%2FL)",
                "[PRL(ng/mL)3](https://www.contemporaryobgyn.net/view/hormone-levels-and-pcos)",
                "[Vit D3 (ng/mL)1](https://www.uptodate.com/contents/vitamin-d-deficiency-beyond-the-basics#:~:text=%E2%97%8FA%20normal%20level%20of,30%20to%2050%20nmol%2FL)",
                "[Vit D3 (ng/mL)2](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6467740/)",
                "[Vit D3 (ng/mL)3](https://www.verywellhealth.com/vitamin-d-more-than-just-a-vitamin-2616313#:~:text=Vitamin%20D%20deficiency%20might%20exacerbate,%2C%20weight%20gain%2C%20and%20anxiety)",
                "[PRG(ng/mL)](https://www.urmc.rochester.edu/encyclopedia/content.aspx?ContentTypeID=167&ContentID=progesterone)",
                "[RBS(mg/dl)1](https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451)",
                "[RBS(mg/dl)2](https://www.cdc.gov/diabetes/basics/pcos.html#:~:text=Diabetes%E2%80%94more%20than%20half%20of,and%20risk%20increases%20with%20age)",
                "[Cycle(R/I)](https://www.jeanhailes.org.au/health-a-z/pcos/irregular-periods-management-treatment#:~:text=Although%20some%20women%20with%20PCOS,be%20irregular%2C%20or%20stop%20altogether.)",
                "[Cycle length(days)](https://www.jeanhailes.org.au/health-a-z/pcos/irregular-periods-management-treatment#:~:text=If%20you%20have%20PCOS%2C%20your,fewer%20menstrual%20cycles%20per%20year)",
                "[Pregnant(Y/N)](https://www.nichd.nih.gov/health/topics/pcos/more_information/FAQs/pregnancy#:~:text=Pregnancy%20complications%20related%20to%20PCOS,as%20are%20women%20without%20PCOS.&text=Some%20research%20shows%20that%20metformin,in%20pregnant%20women%20with%20PCOS.)",
                "[No. of abortions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7744738/#:~:text=In%20addition%2C%20patients%20with%20PCOS,of%2058%25%20(7).)",
                "[hair growth(Y/N)](https://www.healthline.com/health/pcos-hair-loss-2#_noHeaderPrefixedContent)",
                "[Skin darkening (Y/N)](https://www.sepalika.com/pcos/pcos-symptom/pcos-dark-skin-patches/#:~:text=Apart%20from%20cystic%20acne%2C%20hirsutism,commonly%20seen%20in%20skin%20folds.)",
                "[Hair loss(Y/N)](https://www.healthline.com/health/pcos-hair-loss-2)",
                "[Pimples(Y/N)](https://www.medicalnewstoday.com/articles/pcos-acne#diagnosis)",
                "[Fast food (Y/N)](https://www.ccrmivf.com/news-events/food-pcos/#:~:text=%E2%80%9CWomen%20with%20PCOS%20should%20avoid,meats%20like%20fast%20food%20hamburgers)",
                "[Weight gain(Y/N)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6734597/#:~:text=In%20women%20who%20are%20genetically,are%20either%20overweight%20or%20obese.)",
                "[Reg.Exercise(Y/N)](https://exerciseright.com.au/best-types-exercise-pcos/#:~:text=Moderate%20exercise%20like%20brisk%20walking,disease%20and%20type%202%20diabetes.)",
                "[Endometrium (mm)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3283042/#:~:text=Also%2C%20our%20study%20showed%20that,associated%20with%20the%20endometrial%20disease.)",
                "[Avg. Follicle No.1](https://obgyn.onlinelibrary.wiley.com/doi/10.1002/uog.13402)",
                "[Avg. Follicle No.2](https://www.fertilityfamily.co.uk/blog/how-many-eggs-per-follicle-everything-you-need-to-know/#:~:text=A%20woman%20is%20considered%20to,reserve%20is%20greater%20than%2012.)",
                "[Avg. F size (mm)1](https://obgyn.onlinelibrary.wiley.com/doi/10.1002/uog.13402)",
                "[Avg. F size (mm)2](https://www.intechopen.com/chapters/45102#:~:text=A%20normal%20ovary%20consists%20of,are%20known%20as%20dominant%20follicles.)",
                "[Classifiers](https://www.datasciencecentral.com/alternatives-to-logistic-regression/)"
                ]
    s = ''
    if sources:
        for i in links:
            s += "- " + i + "\n"
        st.sidebar.markdown(s)

#---Feature Engineering---------------------------------------------------------------------------------------

with tab2:

    cat_feat = ['Blood Group',
     'Cycle(R/I)',
     'Pregnant(Y/N)',
     'Weight gain(Y/N)',
     'hair growth(Y/N)',
     'Skin darkening (Y/N)',
     'Hair loss(Y/N)',
     'Pimples(Y/N)',
     'Fast food (Y/N)',
     'Reg.Exercise(Y/N)']

    for i in cat_feat:

        df4[i] = df4[i].astype(str)
        df4_corr[i] = df4_corr[i].astype(str)

        df6 = pd.get_dummies(df4)
        df6_corr = pd.get_dummies(df4_corr)

        pcos_var1 = df4['PCOS (Y/N)'].replace(['Yes', 'No'], [1, 0], inplace=True)
        pcos_var2 = df4_corr['PCOS (Y/N)']

        df7 = pd.concat([pcos_var1, df6], axis=1)
        df7_corr = pd.concat([pcos_var2, df6_corr], axis=1)


    st.write("One-hot encoding for categorical variables, besides the target variable, is applied with the MinMax Scaler to be selected as the suggested scaler for non-categorical variables. MinMax Scaler is preferred here to avoid negative values.")
    
    data_show3 = st.checkbox('Show/hide data for reduction')
    if data_show3:
        st.dataframe(df7.head(5).style.set_properties(**{'background-color': 'yellow'}, subset=['PCOS (Y/N)']))
    
    tab2_1, tab2_2 = st.tabs(["via Correlation", "via PCA"])

#---Weakest Corr / Lowest Variances---------------------------------------------------------------------------------------

    with tab2_1:

        st.write("Based on the EDA, 'No. of abortions' is heavily skewed to 0 and does not have a strong relationship with PCOS presence. The 'Weight (Kg)' variable may produce multicollinearity since it is highly correlated with 'Weight Gain(Y/N)' and with 'Height(Cm)'. Hence, 'No. of abortions' and 'Weight (Kg)' are manually removed below.")

        with st.form("key3"):
            df8 = df7.drop(['No. of abortions', 'Weight (Kg)'], axis=1)
            corr_df8 = df8.corr()
            corr_thresh = st.slider("Select minimum correlation magnitude threshold", min_value=0.0, step=0.01, max_value=1.0, value=0.30)
            selected_corr2 = pd.DataFrame(corr_df8['PCOS (Y/N)'])
            selected_corr2.reset_index(inplace=True)
            selected_corr3 = selected_corr2.where(abs(selected_corr2['PCOS (Y/N)']) >= corr_thresh).dropna()
            soc2 = selected_corr3.rename(columns = {'index':'Feature', 'PCOS (Y/N)':'Corr with PCOS (Y/N)'})
            filtered_corr_vars2 = soc2['Feature'].unique().tolist()
            elected_corr_data2 = df8[filtered_corr_vars2]
            elected_var_data2 = pd.DataFrame(df8[filtered_corr_vars2].var())
            elected_var_data2.reset_index(inplace=True)
            soc3 = elected_var_data2.rename(columns = {'index':'Feature', 0:'Variance'})
            soc4 = pd.concat([soc2.set_index('Feature'),soc3.set_index('Feature')], axis=1, join='inner')
            st.write('0.3 is the default minimum strength of correlation coefficent chosen since the strongest ones are below 0.7.')
            button3 = st.form_submit_button("Apply minimum threshold")
        st.dataframe(soc4.sort_values(by=['Corr with PCOS (Y/N)'], key=abs, ascending=False))

        st.write("Only these features will be used for ML in the Classification tab if this dimensionality reduction method is chosen.")

        data_show4 = st.checkbox('Show/hide reduced data via correlation')
        if data_show4:
            st.dataframe(elected_corr_data2.head(5).style.set_properties(**{'background-color': 'yellow'}, subset=['PCOS (Y/N)']))

#---PCA---------------------------------------------------------------------------------------

    with tab2_2:

        st.write("Variables which were manually removed due to multicollinearity since this will part of the PCA work to be done below.")

        Y_pca = df7['PCOS (Y/N)']
        X_pca = df7.drop('PCOS (Y/N)',axis=1)
        max_cols = X_pca.shape[1]

        with st.form("key4"):
            ncomp = st.slider("Select number of PCA components", min_value=1, step=1, max_value=max_cols, value=16)
            pca = PCA(n_components=ncomp,random_state=42)
            pca.fit(X_pca)

            comp_check = pca.explained_variance_ratio_
            comp_check_df = pd.DataFrame(comp_check)
            comp_check_df.reset_index(inplace=True)
            comp_check_df = comp_check_df.rename(columns = {'index':'PC', 0:'Variance'})
            comp_check_df['PC'] = comp_check_df['PC']+1

            pca_eigen = pca.explained_variance_
            pca_eigen_df = pd.DataFrame(pca_eigen)
            pca_eigen_df.reset_index(inplace=True)
            pca_eigen_df = pca_eigen_df.rename(columns = {'index':'PC', 0:'Eigenvalues'})
            pca_eigen_df['PC'] = pca_eigen_df['PC']+1

            X_newpca = pd.DataFrame(pca.fit_transform(X_pca))
            for i in range(0, ncomp):
                X_newpca.rename( {i : "PC{}".format(i+1)}, axis="columns", inplace=True)
            button4 = st.form_submit_button("Apply PCA")
            st.write("{} components can explain around {}% of the variability in the data above.".format(ncomp,round(comp_check.sum()*100,4)))
        

        fig_pca1 = px.line(x=pca_eigen_df['PC'],y=pca_eigen_df['Eigenvalues'], title='PCA Eigenvalues', markers=True).update_layout(title_x=0.5, xaxis_title="Number of Features/Components", yaxis_title="Eigenvalues")
        fig_pca1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },font=dict(
            size=12,
            ),autosize=False,
            width=800,
            height=400,

        )
        fig_pca1.update_traces(line_color='grey', line_width=1)
        fig_pca1.add_shape(type='line',
                x0=0,
                y0=1,
                x1=ncomp,
                y1=1,
                line=dict(color='blue',dash='dash'),
                xref='x',
                yref='y'
        )
        st.write(fig_pca1)

        var = pd.DataFrame(np.cumsum(np.round(comp_check_df['Variance'],decimals=3)*100))
        var.reset_index(inplace=True)
        var_df = var.rename(columns = {'index':'PC'})
        var_df['PC'] = var_df['PC']+1

        fig_pca2 = px.line(x=var_df['PC'],y=var_df['Variance'], title='PCA Variability', markers=True).update_layout(title_x=0.5, xaxis_title="Number of Features/Components", yaxis_title="Variance Explained", yaxis_range=[min(var),100])
        fig_pca2.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },font=dict(
            size=12,
            ),autosize=False,
            width=800,
            height=400,

        )
        fig_pca2.update_traces(line_color='grey', line_width=1)
        fig_pca2.add_shape(type='line',
                x0=0,
                y0=95,
                x1=ncomp,
                y1=95,
                line=dict(color='red',dash='dash'),
                xref='x',
                yref='y'
        )
        st.write(fig_pca2)

        st.write("The threshold set for the percentage of variability explained in the original data is 95%. We will get this with around 16, from 50, components. This is what will be used for ML in the Classification tab if this dimensionality reduction method is chosen.")

        col6, col7 = st.columns((4,8))

        with col6:
            data_show4 = st.checkbox('Show/hide variabilities')
            if data_show4:
                st.dataframe(comp_check_df)

        with col7:
            data_show5 = st.checkbox('Show/hide reduced data via PCA')
            if data_show5:
                st.dataframe(X_newpca)

#---Classification---------------------------------------------------------------------------------------

with tab3:

    with st.form("key_class"):
        Dim_Red = ["Reduced via PCA",
                   "Reduced via Correlation",
                   "Not Reduced"   
                   ]
        ML_data = st.selectbox("Which data would you like to use for machine learning?", Dim_Red)
        if ML_data == "Not Reduced":
            Y = Y_pca
            X = X_pca
        if ML_data == "Reduced via PCA":
            Y = Y_pca
            X = X_newpca
        if ML_data == "Reduced via Correlation":
            Y = elected_corr_data2['PCOS (Y/N)']
            X = elected_corr_data2.drop('PCOS (Y/N)',axis=1)

        test_percent = st.number_input('Input percentage of test data for Train-Test-Split', min_value=0.00, max_value=1.00, step=0.01, value=0.30)
        if test_percent>0.00 and test_percent<1.00:
            st.write("{} of the data will be for training and {} will be for testing.".format(round((1-test_percent)*len(X)),round(test_percent*len(X))))
        elif test_percent == 0.00:
            st.error("A portion of the dataset is needed for testing. Please increase the number above.")
            st.stop()
        else:
            st.error("A portion of the dataset is needed for training. Please decrease the number above.")
            st.stop()

        kchoices = [5,10]
        k = st.selectbox("Select K for K-Fold Cross Validation",kchoices)
        kfold = KFold(n_splits=k, random_state=0, shuffle=True)
        kfold_5 = KFold(n_splits=5, random_state=0, shuffle=True)
        kfold_10 = KFold(n_splits=10, random_state=0, shuffle=True)

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=test_percent, random_state=0)
        X_train_nn, X_test_nn, Y_train_nn, Y_test_nn = train_test_split(X, Y, stratify=Y, random_state=1)
        class_names = ["without PCOS", "with PCOS"]

        button_class = st.form_submit_button("Apply data")

    st.write('These five classifiers were chosen due to the nature of the dataset which has several significant categorical variables. Results for the Train-Test-Split method and the K-Fold CV method are shown below. The hyperparameters are customizable, but the suggested ones are searched via Randomized Search CV and mentioned befor the plots below.')
    

#---Logistic Regression---------------------------------------------------------------------------------------

    with st.expander("Logistic Regression"):
        with st.form("key5"):
            max_iter = st.number_input("Input maximum number of iterations", 10000, 12000, value=10741, step=1, key="max_iter")
            C = st.number_input("Input regularization parameter", 1, 20, value=8, step=1, key="C_LR")
            
            clf1 = LogisticRegression(C=C, max_iter=max_iter)
            clf1.fit(X_train, Y_train)

            Y_predtrain1 = clf1.predict(X_train)
            Y_predtest1 = clf1.predict(X_test)

            train_acc1 = accuracy_score(Y_train, Y_predtrain1)
            test_acc1 = accuracy_score(Y_test, Y_predtest1)

            button5 = st.form_submit_button("Apply hyperparameters")

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l4, m4, r4 = st.columns((3,4,2))
        with m4:
            st.subheader("Train-Test-Split Results")

        col8, col9 = st.columns((4,8))
        with col8:
            st.write("The training accuracy is {}%.".format(train_acc1.round(2)*100))
            st.write("The precision score is {}%.".format(precision_score(Y_test, Y_predtest1, labels=class_names).round(2)*100))
        with col9:
            st.write("The test accuracy is {}%.".format(test_acc1.round(2)*100))
            st.write("The recall score is {}%.".format(recall_score(Y_test, Y_predtest1, labels=class_names).round(2)*100))

        C_sim = list(range(1, 21))
        max_iter_sim = list(range(10000, 12001))
        param_grid1 = dict(C=C_sim,max_iter=max_iter_sim)
        lr = LogisticRegression()
        search_lr = RandomizedSearchCV(estimator=lr, param_distributions=param_grid1, cv = 5, n_jobs=-1)
        search_result_lr = search_lr.fit(X_train, Y_train)
        st.write("The best score of {}% is found when the parameters are {}".format(round(search_result_lr.best_score_*100,2), search_result_lr.best_params_))

        cm1, prc1 = st.columns(2,gap='small')
        with cm1:
            st.markdown("##### Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(clf1, X_test, Y_test, display_labels=class_names, cmap='Blues')
            st.pyplot()
        with prc1:
            st.markdown("##### Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(clf1, X_test, Y_test)
            st.pyplot()

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l5, m5, r5 = st.columns((2,5,2))
        with m5:
            st.subheader("{}-Fold Cross Validation Result".format(k))
        kfold_results1 = cross_val_score(clf1, X, Y, cv=kfold)
        st.write("Accuracy: %.3f%% (%.3f%%)" % (kfold_results1.mean()*100.0, kfold_results1.std()*100.0))

        kfold_results1_5 = cross_val_score(clf1, X, Y, cv=kfold_5)
        kfold_results1_10 = cross_val_score(clf1, X, Y, cv=kfold_10)
        
#---Random Forest---------------------------------------------------------------------------------------

    with st.expander("Random Forest"):
        with st.form("key6"):
            numEstimators = st.number_input("Input number of estimators", 60, 130, value=124, step=1, key="numEstimators")
            max_depth_rf = st.number_input("Input max_depth parameter", min_value=1, max_value=10, value=9, step=1, key="md_rf")
            
            clf2 = RandomForestClassifier(max_depth=max_depth_rf, n_estimators=numEstimators, random_state=0)
            clf2.fit(X_train, Y_train)

            Y_predtrain2 = clf2.predict(X_train)
            Y_predtest2 = clf2.predict(X_test)

            train_acc2 = accuracy_score(Y_train, Y_predtrain2)
            test_acc2 = accuracy_score(Y_test, Y_predtest2)

            button6 = st.form_submit_button("Apply hyperparameters")

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l6, m6, r6 = st.columns((3,4,2))
        with m6:
            st.subheader("Train-Test-Split Results")

        col10, col11 = st.columns((4,8))
        with col10:
            st.write("The training accuracy is {}%.".format(train_acc2.round(2)*100))
            st.write("The precision score is {}%.".format(precision_score(Y_test, Y_predtest2, labels=class_names).round(2)*100))
        with col11:
            st.write("The test accuracy is {}%.".format(test_acc2.round(2)*100))
            st.write("The recall score is {}%.".format(recall_score(Y_test, Y_predtest2, labels=class_names).round(2)*100))
        
        numEstimators_sim = list(range(60, 131))
        max_depth_rf_sim = list(range(1, 11))
        param_grid2 = dict(max_depth=max_depth_rf_sim,n_estimators=numEstimators_sim)
        rf = RandomForestClassifier(random_state=0)
        search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_grid2, cv = 5, n_jobs=-1)
        search_result_rf = search_rf.fit(X_train, Y_train)
        st.write("The best score of {}% is found when the parameters are {}".format(round(search_result_rf.best_score_*100,2), search_result_rf.best_params_))

        cm2, prc2 = st.columns(2,gap='small')
        with cm2:
            st.markdown("##### Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(clf2, X_test, Y_test, display_labels=class_names, cmap='Blues')
            st.pyplot()
        with prc2:
            st.markdown("##### Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(clf2, X_test, Y_test)
            st.pyplot()

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l7, m7, r7 = st.columns((2,5,2))
        with m7:
            st.subheader("{}-Fold Cross Validation Result".format(k))
        kfold_results2 = cross_val_score(clf2, X, Y, cv=kfold)
        st.write("Accuracy: %.3f%% (%.3f%%)" % (kfold_results2.mean()*100.0, kfold_results2.std()*100.0))

        kfold_results2_5 = cross_val_score(clf2, X, Y, cv=kfold_5)
        kfold_results2_10 = cross_val_score(clf2, X, Y, cv=kfold_10)

#---Decision Tree---------------------------------------------------------------------------------------

    with st.expander("Decision Tree"):
        with st.form("key7"):
            crit = st.selectbox("Select criterion", ["entropy","gini","log_loss"])
            max_depth_dtree = st.number_input("Input max_depth parameter", min_value=1, max_value=30, value=12, step=1, key="md_dtree")

            clf3 = DecisionTreeClassifier(criterion=crit, max_depth=max_depth_dtree, random_state=0)
            clf3.fit(X_train, Y_train)

            Y_predtrain3 = clf3.predict(X_train)
            Y_predtest3 = clf3.predict(X_test)

            train_acc3 = accuracy_score(Y_train, Y_predtrain3)
            test_acc3 = accuracy_score(Y_test, Y_predtest3)

            button7 = st.form_submit_button("Apply hyperparameters")

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l8, m8, r8 = st.columns((3,4,2))
        with m8:
            st.subheader("Train-Test-Split Results")

        col12, col13 = st.columns((4,8))
        with col12:
            st.write("The training accuracy is {}%.".format(train_acc3.round(2)*100))
            st.write("The precision score is {}%.".format(precision_score(Y_test, Y_predtest3, labels=class_names).round(2)*100))
        with col13:
            st.write("The test accuracy is {}%.".format(test_acc3.round(2)*100))
            st.write("The recall score is {}%.".format(recall_score(Y_test, Y_predtest3, labels=class_names).round(2)*100))
        
        crit_sim = ["gini","entropy","log_loss"]
        max_depth_dtree_sim = list(range(1, 31))
        param_grid3 = dict(criterion=crit_sim, max_depth=max_depth_dtree_sim)
        dt = DecisionTreeClassifier(random_state=0)
        search_dt = RandomizedSearchCV(estimator=dt, param_distributions=param_grid3, cv = 5, n_jobs=-1)
        search_result_dt = search_dt.fit(X_train, Y_train)
        st.write("The best score of {}% is found when the parameters are {}".format(round(search_result_dt.best_score_*100,2), search_result_dt.best_params_))

        cm3, prc3 = st.columns(2,gap='small')
        with cm3:
            st.markdown("##### Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(clf3, X_test, Y_test, display_labels=class_names, cmap='Blues')
            st.pyplot()
        with prc3:
            st.markdown("##### Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(clf3, X_test, Y_test)
            st.pyplot()

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l9, m9, r9 = st.columns((2,5,2))
        with m9:
            st.subheader("{}-Fold Cross Validation Result".format(k))
        kfold_results3 = cross_val_score(clf3, X, Y, cv=kfold)
        st.write("Accuracy: %.3f%% (%.3f%%)" % (kfold_results3.mean()*100.0, kfold_results3.std()*100.0))

        kfold_results3_5 = cross_val_score(clf3, X, Y, cv=kfold_5)
        kfold_results3_10 = cross_val_score(clf3, X, Y, cv=kfold_10)

#---KNeighbors---------------------------------------------------------------------------------------

    with st.expander("KNN"):
        with st.form("key8"):
            nn = st.number_input("Input the number of nearest neighbors", min_value=1, max_value=15, value=9, step=2, key="knn")

            clf4 = KNeighborsClassifier(n_neighbors=nn)
            clf4.fit(X_train, Y_train)

            Y_predtrain4 = clf4.predict(X_train)
            Y_predtest4 = clf4.predict(X_test)

            train_acc4 = accuracy_score(Y_train, Y_predtrain4)
            test_acc4 = accuracy_score(Y_test, Y_predtest4)

            button8 = st.form_submit_button("Apply hyperparameters")

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l10, m10, r10 = st.columns((3,4,2))
        with m10:
            st.subheader("Train-Test-Split Results")

        col14, col15 = st.columns((4,8))
        with col14:
            st.write("The training accuracy is {}%.".format(train_acc4.round(2)*100))
            st.write("The precision score is {}%.".format(precision_score(Y_test, Y_predtest4, labels=class_names).round(2)*100))
        with col15:
            st.write("The test accuracy is {}%.".format(test_acc4.round(2)*100))
            st.write("The recall score is {}%.".format(recall_score(Y_test, Y_predtest4, labels=class_names).round(2)*100))
        
        nn_sim = list(range(1, 16))
        param_grid4 = dict(n_neighbors=nn_sim)
        knn = KNeighborsClassifier()
        search_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_grid4, cv = 5, n_jobs=-1)
        search_result_knn = search_knn.fit(X_train, Y_train)
        st.write("The best score of {}% is found when the parameters are {}".format(round(search_result_knn.best_score_*100,2), search_result_knn.best_params_))

        cm4, prc4 = st.columns(2,gap='small')
        with cm4:
            st.markdown("##### Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(clf4, X_test, Y_test, display_labels=class_names, cmap='Blues')
            st.pyplot()
        with prc4:
            st.markdown("##### Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(clf4, X_test, Y_test)
            st.pyplot()

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l11, m11, r11 = st.columns((2,5,2))
        with m11:
            st.subheader("{}-Fold Cross Validation Result".format(k))
        kfold_results4 = cross_val_score(clf4, X, Y, cv=kfold)
        st.write("Accuracy: %.3f%% (%.3f%%)" % (kfold_results4.mean()*100.0, kfold_results4.std()*100.0))

        kfold_results4_5 = cross_val_score(clf4, X, Y, cv=kfold_5)
        kfold_results4_10 = cross_val_score(clf4, X, Y, cv=kfold_10)

#---Neural Networks---------------------------------------------------------------------------------------

    with st.expander("Neural Network"):
        X_train_nn, X_test_nn, Y_train_nn, Y_test_nn = train_test_split(X, Y, stratify=Y, random_state=1)
        with st.form("key9"):
            max_iter_nn = st.number_input("Input maximum number of iterations", 100, 3000, value=555, step=1, key="max_iter_nn")

            clf5 = MLPClassifier(max_iter=max_iter_nn, random_state=0)
            clf5.fit(X_train_nn, Y_train_nn)

            Y_predtrain5 = clf5.predict(X_train_nn)
            Y_predtest5 = clf5.predict(X_test_nn)

            train_acc5 = accuracy_score(Y_train_nn, Y_predtrain5)
            test_acc5 = accuracy_score(Y_test_nn, Y_predtest5)

            button9 = st.form_submit_button("Apply hyperparameters")

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l12, m12, r12 = st.columns((3,4,2))
        with m12:
            st.subheader("Train-Test-Split Results")

        col16, col17 = st.columns((4,8))
        with col16:
            st.write("The training accuracy is {}%.".format(train_acc5.round(2)*100))
            st.write("The precision score is {}%.".format(precision_score(Y_test_nn, Y_predtest5, labels=class_names).round(2)*100))
        with col17:
            st.write("The test accuracy is {}%.".format(test_acc5.round(2)*100))
            st.write("The recall score is {}%.".format(recall_score(Y_test_nn, Y_predtest5, labels=class_names).round(2)*100))
        

        max_iter_nn_sim = list(range(100, 3001))
        param_grid5 = dict(max_iter=max_iter_nn_sim)
        mlp = MLPClassifier(random_state=0)
        search_mlp = RandomizedSearchCV(estimator=mlp, param_distributions=param_grid5, cv = 5, n_jobs=-1)
        search_result_mlp = search_mlp.fit(X_train_nn, Y_train_nn)
        st.write("The best score of {}% is found when the parameters are {}".format(round(search_result_mlp.best_score_*100,2), search_result_mlp.best_params_))

        cm5, prc5 = st.columns(2,gap='small')
        with cm5:
            st.markdown("##### Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(clf5, X_test_nn, Y_test_nn, display_labels=class_names, cmap='Blues')
            st.pyplot()
        with prc5:
            st.markdown("##### Precision-Recall Curve")
            PrecisionRecallDisplay.from_estimator(clf5, X_test_nn, Y_test_nn)
            st.pyplot()

        st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        l13, m13, r13 = st.columns((2,5,2))
        with m13:
            st.subheader("{}-Fold Cross Validation Result".format(k))
        kfold_results5 = cross_val_score(clf5, X, Y, cv=kfold)
        st.write("Accuracy: %.3f%% (%.3f%%)" % (kfold_results5.mean()*100.0, kfold_results5.std()*100.0))
        
        kfold_results5_5 = cross_val_score(clf5, X, Y, cv=kfold_5)
        kfold_results5_10 = cross_val_score(clf5, X, Y, cv=kfold_10)

#---Classifier Comparison---------------------------------------------------------------------------------------

    with st.expander("Comparison"):
        st.markdown('The line chart below shows the performances of the different estimators when applied to the training dataset, test dataset, 5-fold cross-validated dataset, and 10-fold cross-validated dataset. For the train-test-split algorithm, the program follows the percentage split set above, but the default is a 70-30% split. With this split after applying minimum-maximum scaler and PCA to reduce dimensionality and the optimum hyperparameters per method, it can be seen below that the training accuracy is generally higher than the testing accuracy. The highest training accouracy is 100% via Decision Tree while the highest testing accuracy is 89.7% via Neural Networks. Despite having the highest training accuracy, the Decision Tree classifier also has the lowest testing accuracy of 80.4%. The smallest ratios in train accuracy and test accuracy is found via the KNN classifier with the test accuracy surpassing the training accuracy.')
        st.markdown('In terms of the K-Fold cross validation, the results for 5-fold and 10-fold are nearly the same with the 10-fold algorithm surpassing the 5-fold algorithm for all estimators.')
        st.markdown('The Train-Test-Split algorithm gave highest test accuracies for all classifiers except for the KNN and Neural Network, while K-Fold CV classifiers performed better for the Tree methods. For the Logistic Regression, accuracies are close to each other, but the 10-Fold CV estimator gives the highest one, not counting the training accuracy.')
        st.markdown('The two best models for me would be applying the Logistic Regression to the 10-Fold CV dataset and the Neural Network classifiers to the Train-Test-Split dataset. If the dimensionality reduction is switched to via Correlation and the optimum hyperparameters is applied, we will see that even though the training accuracies are lower, these two estimators applied to the Test-Train-Split method will still give us the best test accuracies.')
        methods = ['Logistic', 'Rand Forest', 'Dtree', 'KNN', 'Neural Network']
        accs = {'Training': [train_acc1, train_acc2, train_acc3, train_acc4, train_acc5], 
                'Test': [test_acc1, test_acc2, test_acc3, test_acc4, test_acc5],
                '5-Fold CV': [kfold_results1_5.mean(), kfold_results2_5.mean(), kfold_results3_5.mean(), kfold_results4_5.mean(), kfold_results5_5.mean()],
                '10-Fold CV': [kfold_results1_10.mean(), kfold_results2_10.mean(), kfold_results3_10.mean(), kfold_results4_10.mean(), kfold_results5_10.mean()],
                }
        accs_df = pd.DataFrame(data=accs)
        fig_acc = px.line(accs_df, x=methods, y=['Training', 'Test', '5-Fold CV', '10-Fold CV'], title='Data Splitting vs. Classifier Accuracy Comparison').update_layout(title_x=0.5, xaxis_title="Estimator/Classifier", yaxis_title="Accuracy", yaxis_range=[0.5,1])
        fig_acc.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        },font=dict(
            # family="Courier New, monospace",
            size=12,
            # color="RebeccaPurple"
            ),autosize=False,
            width=650,
            height=500,
        )
        st.write(fig_acc)
        fsd = accs_df['Training']/accs_df['Test']

    