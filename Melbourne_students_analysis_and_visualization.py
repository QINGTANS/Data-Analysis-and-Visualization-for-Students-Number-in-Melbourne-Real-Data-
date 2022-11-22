import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
from numpy import arange
from scipy.stats import shapiro
from scipy.stats import wilcoxon

'''

data pre-processing for population information csv file

'''

# making data frame from csv file
population_data = pd.read_csv("City_of_Melbourne_Population_Forecasts_2016_to_2041_-_Age_and_Gender.csv", index_col="Year")

# retrieving row by loc method
for i in range(2021, 2029):
    df = population_data.loc[i]
    df = df[
        ["Geography", "Total population", "Total 5-9 years", "Total 10-14 years", "Total 15-19 years", "Male 5-9 years",
         "Male 10-14 years", "Male 15-19 years", "Female 5-9 years", "Female 10-14 years", "Female 15-19 years"]]
    if i == 2021:
        df1 = df
    else:
        df1 = pd.concat([df1, df], axis=0)

# calcute primary/secondary for total
age1 = list(df1["Total 5-9 years"])

age2 = list(df1["Total 10-14 years"])
ave2 = []
for num in age2:
    ave2.append(num / 5)

age3 = list(df1["Total 15-19 years"])
ave3 = []
for num in age3:
    ave3.append(num / 5)

primary = []
secondary = []
for i in range(0, len(age1)):
    primary.append(age1[i] + 2 * ave2[i])
    secondary.append(3 * ave2[i] + 3 * ave3[i])

df1["Primary"] = primary
df1["Secondary"] = secondary

# calcute primary/secondary for male
mAge1 = list(df1["Male 5-9 years"])
mAge2 = list(df1["Male 10-14 years"])
mAve2 = []
for num in mAge2:
    mAve2.append(num / 5)

mAge3 = list(df1["Male 15-19 years"])
mAve3 = []
for num in mAge3:
    mAve3.append(num / 5)

mPrimary = []
mSecondary = []
for i in range(0, len(mAge1)):
    mPrimary.append(mAge1[i] + 2 * mAve2[i])
    mSecondary.append(3 * mAve2[i] + 3 * mAve3[i])

df1["Male primary"] = mPrimary
df1["Male secondary"] = mSecondary

# calcute primary/secondary for female
fAge1 = list(df1["Female 5-9 years"])
fAge2 = list(df1["Female 10-14 years"])
fAve2 = []
for num in fAge2:
    fAve2.append(num / 5)

fAge3 = list(df1["Female 15-19 years"])
fAve3 = []
for num in fAge3:
    fAve3.append(num / 5)

fPrimary = []
fSecondary = []
for i in range(0, len(fAge1)):
    fPrimary.append(fAge1[i] + 2 * fAve2[i])
    fSecondary.append(3 * fAve2[i] + 3 * fAve3[i])

df1["Female primary"] = fPrimary
df1["Female secondary"] = fSecondary

df1.to_csv("population.csv")

'''

data pre-processing for school location and enrollment information, and merge them together to data.csv

'''

data309=pd.read_csv('dv309_schoollocations2021.csv',encoding = 'ISO-8859-1')
data316=pd.read_csv("dv316-allschoolsFTEenrolmentsFeb2021.csv",encoding = 'ISO-8859-1')

location=data309.iloc[:,[2,3,8,19,20]]
enrolment=data316.iloc[:,6:53]
enrolment.insert(0,"SCHOOL_NO",data316.iloc[:,2])
enrolment.insert(1,"School_Name",data316.iloc[:,3])
#merge school and location
school=pd.merge(location, enrolment,on=["SCHOOL_NO","School_Name"],how='inner')
#read population data
population=pd.read_csv('population.csv',encoding = 'ISO-8859-1')
#remove () and compute population in each suburb
pattern = r" [\(（][^\)）]+[\)）]"
for i in population.index:
    population.iloc[i,1] = population.iloc[i,1].casefold()
    population.iloc[i,1] = re.sub(pattern,"",population.iloc[i,1])

for i in school.index:
    school.iloc[i,2]=school.iloc[i,2].casefold()

for i in range(len(population)-1):
    if population.iloc[i,1] == population.iloc[i+1,1]:
        for j in range(2,len(population.columns)):
            population.iloc[i,j] = population.iloc[i,j]+population.iloc[i+1,j]

for i in range(0,8):
    population = population.drop(index=[i*14+3])
    population = population.drop(index=[i*14+6])
    population = population.drop(index=[i*14+13])
#merge school and population
school = school.merge(population.iloc[0:12,1],left_on="Address_Town",right_on="Geography",how="inner")
school.to_csv('school_visualization_prepare.csv', index=False)
population.to_csv("population_visualization_prepare.csv", index=False)
#save data
data_merge = pd.merge(left=population,right=school,left_on="Geography",right_on="Address_Town")
data_merge = data_merge.drop(["Geography_y"], axis=1)
data_merge = data_merge.rename(columns={'Geography_x':'Geography'})
data_merge.to_csv("data.csv", index=False)

'''

data pre-processing for enrollment information from 

'''

data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')
data2017 = pd.read_csv('dv245-fteenrolmentsfeb2017.csv', encoding = 'ISO-8859-1')
data2018 = pd.read_csv('dv272-fteenrolmentsfeb2018.csv', encoding = 'ISO-8859-1')
data2019 = pd.read_csv('dv290-allschoolsFTEenrolmentsFeb2019.csv', encoding = 'ISO-8859-1')
data2020 = pd.read_csv('dv300-allschoolsFTEenrolmentsFeb2020.csv', encoding = 'ISO-8859-1')
school2021 = data.loc[data['Year'] == 2021, ['Geography', 'SCHOOL_NO', 'School_Name', 'Primary Total', 'Secondary Total']]
school2021.reset_index(drop=True, inplace=True)
school2021.columns = ['Suburb', 'SCHOOL_NO', 'School_Name', 'Primary 2021', 'Secondary 2021']
school2017 = data2017.iloc[:,[2,3,30,52]]
school2017.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2017', 'Secondary 2017']
school2018 = data2018.iloc[:,[2,3,30,52]]
school2018.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2018', 'Secondary 2018']
school2019 = data2019.iloc[:,[2,3,30,52]]
school2019.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2019', 'Secondary 2019']
school2020 = data2020.iloc[:,[2,3,30,52]]
school2020.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2020', 'Secondary 2020']
school1 = pd.merge(school2017, school2021, on=["SCHOOL_NO","School_Name"],how='right')
school2 = pd.merge(school2018, school2021, on=["SCHOOL_NO","School_Name"],how='right')
school3 = pd.merge(school2019, school2021, on=["SCHOOL_NO","School_Name"],how='right')
school4 = pd.merge(school2020, school2021, on=["SCHOOL_NO","School_Name"],how='right')
schools = pd.concat([school1,school2,school3,school4], axis = 1)
school_result = schools.iloc[:, [0,1,2,3,9,10,16,17,23,24,25,26,27]]
school_result = school_result.fillna(0)
order = ['SCHOOL_NO', 'School_Name', 'Suburb', 'Primary 2017', 'Primary 2018', 'Primary 2019', 'Primary 2020', 'Primary 2021',
         'Secondary 2017', 'Secondary 2018', 'Secondary 2019', 'Secondary 2020', 'Secondary 2021']
school_result = school_result[order]
school_result.to_csv('school_result.csv', index=False)

'''

data visualization

'''

population = pd.read_csv("population_visualization_prepare.csv")

# store different regions
block_list = []

for i in range(0, len(population)):
    block = population.iloc[i]['Geography']
    if block not in block_list and block != 'city of melbourne':
        block_list.append(block)

# plot drawing: primary students number in next few years
legend_information = []
for block in block_list:
    legend_information.append(block.upper())
    block_primary_series = population[population['Geography'] == block]['Primary']
    year = population[population['Geography'] == block]['Year']
    plt.plot(year, block_primary_series.values)

plt.title("Primary Students Number in Next Few Years")
plt.xlabel("year", fontsize=13)
plt.ylabel("students", fontsize=13)
plt.ylim(-50, 2250)
plt.legend(legend_information, framealpha=0.3)

plt.savefig('primary_students_in_years.png', dpi=200, bbox_inches='tight')
plt.close()


# plot drawing: secondary students number in next few years
legend_information1 = []
for block in block_list:
    legend_information1.append(block.upper())
    block_primary_series = population[population['Geography'] == block]['Secondary']
    year = population[population['Geography'] == block]['Year']
    plt.plot(year, block_primary_series.values)

plt.title("Secondary Students Number in Next Few Years")
plt.xlabel("year", fontsize=13)
plt.ylabel("students", fontsize=13)
plt.legend(legend_information1, framealpha=0.3, loc='upper left')

plt.savefig('secondary_students_in_years.png', dpi=200, bbox_inches='tight')
plt.close()

# some little change on school.csv for visualization
education_sector = []
school_type = []
school_location = pd.read_csv("dv309_schoollocations2021.csv", encoding = 'ISO-8859-1')
school = pd.read_csv("school_visualization_prepare.csv", encoding = 'ISO-8859-1')
for index, row in school.iterrows():
    school_name = row[1]
    for index1, row1 in school_location.iterrows():
        if row1[3] == school_name:
            education_sector.append(row1[0])
            school_type.append(row1[4])
            break

school.insert(0, 'School_Type', school_type)
school.insert(0, 'Education_Sector', education_sector)
school = school.drop(columns='Geography')
school.to_csv('school_renew.csv', index=False)

# prepare for drawing bar chart
block_name = []
pri_age_list = []
sec_age_list = []
pri_stu_list = []
sec_stu_list = []
for block in block_list:
    block_name.append(block.upper())
    m = population.loc[(population['Geography'] == block) & (population['Year'] == 2021)]
    pri_age_list.append(int(m['Primary'].values))
    sec_age_list.append(int(m['Secondary'].values))
    pri_sum = 0
    sec_sum = 0
    n = school.loc[school['Address_Town'] == block]
    for index, row in n.iterrows():
        pri_sum += row['Primary Total']
        sec_sum += row['Secondary Total']
    pri_stu_list.append(pri_sum)
    sec_stu_list.append(sec_sum)

# drawing two bar charts
plt.bar(arange(len(pri_age_list))-0.3, pri_age_list, width=0.3, label="children in primary age", color='blue')
plt.bar(arange(len(pri_stu_list)), pri_stu_list, width=0.3, label="number of primary students", color='red')
plt.xticks(arange(len(block_name)), block_name, rotation=45)
plt.title("Primary Students and Primary Children for Each Suburb")
plt.xlabel("Suburb", fontsize=14)
plt.ylabel("Children number in 2021", fontsize=12)
plt.legend(loc="upper right")
plt.savefig("regional_primary.png", dpi=200, bbox_inches='tight')
plt.close()

plt.bar(arange(len(sec_age_list))-0.3, sec_age_list, width=0.3, label="children in secondary age", color='blue')
plt.bar(arange(len(sec_stu_list)), sec_stu_list, width=0.3, label="number of secondary students", color='red')
plt.xticks(arange(len(block_name)), block_name, rotation=45)
plt.title("Secondary Students and Secondary Children for Each Suburb")
plt.xlabel("Suburb", fontsize=14)
plt.ylabel("Children number in 2021", fontsize=12)
plt.legend(loc="upper right")
plt.savefig("regional_secondary.png", dpi=200, bbox_inches='tight')
plt.close()

# draw pie charts, the percentage of the three education_sector types in Melbourne
education_types = ['Government', 'Independent', 'Catholic']
pri_student_typenum = []
sec_student_typenum = []
pri_male = []
pri_female = []
sec_male = []
sec_female = []
for education_type in education_types:
    pri_total_num = 0
    sec_total_num = 0
    pri_total_male = 0
    pri_total_female = 0
    sec_total_male = 0
    sec_total_female = 0
    m = school.loc[school['Education_Sector'] == education_type]
    for index, row in m.iterrows():
        pri_total_num += row['Primary Total']
        sec_total_num += row['Secondary Total']
        pri_total_male = pri_total_male + (row['Prep Males Total'] + row['Year 1 Males Total'] +
                        row['Year 2 Males Total'] + row['Year 3 Males Total'] + row['Year 4 Males Total']
                        + row['Year 5 Males Total'] + row['Year 6 Males Total'] + row['Primary Ungraded Males Total'])

        pri_total_female = pri_total_female + (row['Prep Females Total'] + row['Year 1 Females Total'] +
                            row['Year 2 Females Total'] + row['Year 3 Females Total'] + row['Year 4 Females Total']
                            + row['Year 5 Females Total'] + row['Year 6 Females Total'] +
                            row['Primary Ungraded Females Total'])

        sec_total_male = sec_total_male + (row['Year 7 Males Total'] +row['Year 8 Males Total'] +
                        row['Year 9 Males Total'] + row['Year 10 Males Total']+ row['Year 11 Males Total'] +
                        row['Year 12 Males'] + row['Secondary Ungraded Males Total'])

        sec_total_female = sec_total_female + (row['Year 7 Females Total'] +row['Year 8 Females Total'] +
                        row['Year 9 Females Total'] + row['Year 10 Females Total'] + row['Year 11 Females Total'] +
                        row['Year 12 Females'] + row['Secondary Ungraded Females Total'])

    pri_student_typenum.append(pri_total_num)
    sec_student_typenum.append(sec_total_num)
    pri_male.append(pri_total_male)
    pri_female.append(pri_total_female)
    sec_male.append(sec_total_male)
    sec_female.append(sec_total_female)

plt.pie(pri_student_typenum, labels=education_types, autopct="%1.2f%%", colors=['c', 'm', 'y'],
        textprops={'fontsize': 14}, labeldistance=1.05)
plt.legend(loc='lower left', bbox_to_anchor=(0.8, 0.8))
plt.title("Percentage of Primary students in Three School Types (2021)", fontsize=15)
plt.savefig("primary_pie.png", dpi=200, bbox_inches='tight')
plt.close()

plt.pie(sec_student_typenum, labels=education_types, autopct="%1.2f%%", colors=['c', 'm', 'y'],
        textprops={'fontsize': 14}, labeldistance=1.05)
plt.legend(loc='lower left', bbox_to_anchor=(-0.2, 0.8))
plt.title("Percentage of Secondary students in Three School Types (2021)", fontsize=15)
plt.savefig("secondary_pie.png", dpi=200, bbox_inches='tight')
plt.close()

'''

2.1 data analysis for the difference in enrollment between genders

'''
#difference analysis
data=pd.read_csv('data.csv',encoding = 'ISO-8859-1')
loca=pd.DataFrame(np.unique(data.loc[:,'Geography']))
#count enrollment in 2021
pritable=pd.DataFrame(data=0,index=loca[0],columns=('male','female'))

for x in data.index:
    if data.loc[x,'Year']==2021:
        pritable.loc[data.loc[x,'Geography'],'male']=data.loc[x,'Male primary']+pritable.loc[data.loc[x,'Geography'],'male']
        pritable.loc[data.loc[x,'Geography'],'female']=data.loc[x,'Female primary']+pritable.loc[data.loc[x,'Geography'],'female']
sectable=pd.DataFrame(data=0,index=loca[0],columns=('male','female'))

for x in data.index:
    if data.loc[x,'Year']==2021:
        sectable.loc[data.loc[x,'Geography'],'male']=data.loc[x,'Male secondary']+sectable.loc[data.loc[x,'Geography'],'male']
        sectable.loc[data.loc[x,'Geography'],'female']=data.loc[x,'Female secondary']+sectable.loc[data.loc[x,'Geography'],'female']
#Normality test
shapiro(pritable.iloc[:,0])
shapiro(pritable.iloc[:,1])
shapiro(sectable.iloc[:,0])
shapiro(sectable.iloc[:,1])
#Mean
np.mean(pritable.iloc[:,0])
np.mean(pritable.iloc[:,1])
np.mean(sectable.iloc[:,0])
np.mean(sectable.iloc[:,1])
#Std
np.std(pritable.iloc[:,0],ddof=1)
np.std(pritable.iloc[:,1],ddof=1)
np.std(sectable.iloc[:,0],ddof=1)
np.std(sectable.iloc[:,1],ddof=1)
#Wilcoxon test
pri=pritable.iloc[:,0]-pritable.iloc[:,1]
sec=sectable.iloc[:,0]-sectable.iloc[:,1]
wilcoxon(pri)
wilcoxon(sec)

'''

2.2 data analysis for current situation in 2021

'''
# Preprocess datasets from 2017 to 2020
data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')
data2017 = pd.read_csv('dv245-fteenrolmentsfeb2017.csv', encoding = 'ISO-8859-1')
data2018 = pd.read_csv('dv272-fteenrolmentsfeb2018.csv', encoding = 'ISO-8859-1')
data2019 = pd.read_csv('dv290-allschoolsFTEenrolmentsFeb2019.csv', encoding = 'ISO-8859-1')
data2020 = pd.read_csv('dv300-allschoolsFTEenrolmentsFeb2020.csv', encoding = 'ISO-8859-1')
school2021 = data.loc[data['Year'] == 2021, ['Geography', 'SCHOOL_NO', 'School_Name', 'Primary Total', 'Secondary Total']]
school2021.reset_index(drop=True, inplace=True)
school2021.columns = ['Suburb', 'SCHOOL_NO', 'School_Name', 'Primary 2021', 'Secondary 2021']
school2017 = data2017.iloc[:,[2,3,30,52]]
school2017.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2017', 'Secondary 2017']
school2018 = data2018.iloc[:,[2,3,30,52]]
school2018.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2018', 'Secondary 2018']
school2019 = data2019.iloc[:,[2,3,30,52]]
school2019.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2019', 'Secondary 2019']
school2020 = data2020.iloc[:,[2,3,30,52]]
school2020.columns = ['SCHOOL_NO', 'School_Name', 'Primary 2020', 'Secondary 2020']
school1 = pd.merge(school2017, school2021, on=["SCHOOL_NO","School_Name"],how='right')
school2 = pd.merge(school2018, school2021, on=["SCHOOL_NO","School_Name"],how='right')
school3 = pd.merge(school2019, school2021, on=["SCHOOL_NO","School_Name"],how='right')
school4 = pd.merge(school2020, school2021, on=["SCHOOL_NO","School_Name"],how='right')
schools = pd.concat([school1,school2,school3,school4], axis = 1)
school_result = schools.iloc[:, [0,1,2,3,9,10,16,17,23,24,25,26,27]]
school_result = school_result.fillna(0)
order = ['SCHOOL_NO', 'School_Name', 'Suburb', 'Primary 2017', 'Primary 2018', 'Primary 2019', 'Primary 2020', 'Primary 2021',
         'Secondary 2017', 'Secondary 2018', 'Secondary 2019', 'Secondary 2020', 'Secondary 2021']
school_result = school_result[order]



# The primary difference and secondary difference of each suburb in 2021
data = pd.read_csv('data.csv', encoding = 'ISO-8859-1')
geo = (data['Geography'].unique()).tolist()
school = data.loc[data['Year'] == 2021, ['Geography', 'Primary Total', 'Secondary Total']]
school.reset_index(drop=True, inplace=True)
population = data.loc[data['Year'] == 2021, ['Geography', 'Total 5-9 years', 'Total 10-14 years', 'Total 15-19 years']]
population.reset_index(drop=True, inplace=True)
town_school = []
for i in range(len(geo)):
    primary_t = 0
    secondary_t = 0
    for j in range(len(school['Geography'])):
        if geo[i] == school.Geography[j]:
            primary_t += school.iloc[j,1]
            secondary_t += school.iloc[j,2]
    town_school.append([geo[i], primary_t, secondary_t])
town_school = pd.DataFrame(town_school, columns = ['Geography', 'Primary Total', 'Secondary Total'])
town_population = population.drop_duplicates()
town_population.reset_index(drop=True, inplace=True)
Total = []
for i in range(len(geo)):
    if town_school.iloc[i,1] >= town_population.iloc[i,1] + town_population.iloc[i,2]/5*2:
        Total.append([town_school.iloc[i,1] - town_population.iloc[i,1] - town_population.iloc[i,2]/5*2, "Yes"])
    else:
        Total.append([town_school.iloc[i,1] - town_population.iloc[i,1] - town_population.iloc[i,2]/5*2, "No"])
    if town_school.iloc[i,2] >= town_population.iloc[i,2]/5*3 + town_population.iloc[i,3]/5*3:
        Total.append([town_school.iloc[i,2] - town_population.iloc[i,2]/5*3 - town_population.iloc[i,3]/5*3, "Yes"])
    else:
        Total.append([town_school.iloc[i,2] - town_population.iloc[i,2]/5*3 - town_population.iloc[i,3]/5*3, "No"])
total2021 = pd.DataFrame(np.array(Total).reshape(10,4), columns = ['Primary Difference(school-population)', 'Capacity_P', 'Secondary Difference(school-population)', 'Capacity_S'])
total2021.insert(0, 'Suburb', geo)
total2021['Primary Difference(school-population)'] = pd.to_numeric(total2021['Primary Difference(school-population)'])
total2021['Secondary Difference(school-population)'] = pd.to_numeric(total2021['Secondary Difference(school-population)'])
total2021.set_index(['Suburb'], inplace=True)
total2021.to_csv("total2021.csv")



# Find the maximum capacity of schools in each suburb from 2017 to 2021
school_max = []
for i in range(len(school_result['SCHOOL_NO'])):
    school_max.append([school_result['Suburb'][i], school_result.iloc[i,3:8].max(), school_result.iloc[i,8:13].max()])
school_max = pd.DataFrame(school_max, columns = ['Suburb', 'Primary', 'Secondary'])
school_geo = []
for i in range (len(geo)):
    primary_total = 0
    secondary_total = 0
    for j in range(len(school_max['Suburb'])):
        if geo[i] == school_max.Suburb[j]:
            primary_total += school_max.iloc[j,1]
            secondary_total += school_max.iloc[j,2]
    school_geo.append([geo[i], primary_total, secondary_total])
school_geo = pd.DataFrame(school_geo, columns = ['Suburb', 'Primary', 'Secondary'])
school_geo.set_index(['Suburb'], inplace=True)
school_geo.to_csv("school_geography.csv")

'''

2.3 prediction from 2022 to 2028

'''
# create "students vs schools 2021-2028.csv"
data = pd.read_csv("data.csv")
data = data[["Year", "Geography", "Primary", "Secondary", "Primary Total", "Secondary Total"]]

# calculate the schools capacity for each suburbs
s = {}
rows = []
for i in range(0, len(list(data["Geography"]))):
    if list(data["Year"])[i] == 2021:
        if data["Geography"][i] not in s.keys():
            s[data["Geography"][i]] = [data["Primary Total"][i], data["Secondary Total"][i], 1]

        else:
            s[data["Geography"][i]][0] += data["Primary Total"][i]
            s[data["Geography"][i]][1] += data["Secondary Total"][i]
            s[data["Geography"][i]][2] += 1

for k, v in s.items():
    row = [k]

    for i in v:
        row.append(i)
    rows.append(row)

dfSchool = pd.DataFrame(rows, columns=['Geography', 'Primary total', 'Secondary total', 'School total'])

data1 = pd.read_csv("school_geography.csv")

# get the biggest capicity for primary schools
primaryExpect = list(data1["Primary"])
dfSchool["Primary expection"] = primaryExpect

# get the biggest capicity for secondary schools
secondaryExpect = list(data1["Secondary"])
dfSchool["Secondary expection"] = secondaryExpect

dfPopulation = data[["Year", "Geography", "Primary", "Secondary"]]
dfPopulation = dfPopulation.drop_duplicates()

df = pd.merge(dfPopulation, dfSchool, on='Geography', how='outer')

# to get the differences between student number and school capicity
priDifferences = []
for i in range(0, len(list(df["Year"]))):
    priDifferences.append(list(df["Primary expection"])[i] - list(df["Primary"])[i])
df["Primary differences"] = priDifferences

secDifferences = []
for i in range(0, len(list(df["Year"]))):
    secDifferences.append(list(df["Secondary expection"])[i] - list(df["Secondary"])[i])
df["Secondary differences"] = secDifferences

df = df[["Year", "Geography", "Primary", "Primary expection", "Primary differences", "Secondary", "Secondary expection",
         "Secondary differences"]]
df.to_csv("students vs schools 2021-2028.csv")

# visualization for analysis
data = pd.read_csv("students vs schools 2021-2028.csv", index_col="Geography")

i = 0
while i < len(list(data.index)):

    x = []
    primary = []
    secondary = []

    for j in range(0, 8):
        x.append(list(data["Year"])[j + i])
        primary.append(list(data["Primary differences"])[j + i])
        secondary.append(list(data["Secondary differences"])[j + i])

    X_axis = np.arange(len(x))

    plt.bar(X_axis - 0.2, primary, 0.4, label='primary')
    plt.bar(X_axis + 0.2, secondary, 0.4, label='secondary')

    plt.xticks(X_axis, x)

    plt.xlabel("Years")
    plt.ylabel("differences value")
    plt.title(list(data.index)[i])
    plt.legend()

    plt.savefig(str(list(data.index)[i]) + '.png')
    plt.clf()
    i = i + 8