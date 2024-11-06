import altair as alt
import pandas as pd 

# Color palette
color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]

def barchart_proportions(df: pd.DataFrame, x: str, group: str, facet: str, title: str) -> alt.Chart:
    '''
    Function to create a bar chart with proportions and facet
    '''
    # Calculate proportions
    grouped_df: pd.DataFrame = df.groupby(['Family Size', 'Sex', 'Survived']).size().reset_index(name='Count')

    # Step 2: Calculate the total count for each FamilySize and Sex group
    grouped_df['Total'] = grouped_df.groupby(['Family Size', 'Sex'])['Count'].transform('sum')

    # Step 3: Calculate the proportion
    grouped_df['Proportion'] = grouped_df['Count'] / grouped_df['Total']

    # Create the stacked bar chart with proportions
    chart: alt.Chart = alt.Chart(grouped_df).mark_bar().encode(
        x=alt.X(f"{x}:N", title=f"{x}"),
        y=alt.Y('Proportion:Q', title='Proportion', axis=alt.Axis(format='%')),
        color=alt.Color(f"{group}:N", scale=alt.Scale(range=color_list)),
        column=alt.Column(f"{facet}:N", title=f"{facet}")
    ).properties(
        title=title
    )

    return chart


def proportion_chart(df: pd.DataFrame, x: str, group: str, title: str) -> alt.Chart:
    '''
    Function to create a stacked chart without facet
    '''
    # Calculate proportions
    grouped_df: pd.DataFrame = df.groupby([x, group]).size().reset_index(name='Count')

    # Step 2: Calculate the total count for each Family
    grouped_df['Total'] = grouped_df.groupby(x)['Count'].transform('sum')
    grouped_df['Proportion'] = grouped_df['Count'] / grouped_df['Total']

    chart: alt.Chart = alt.Chart(grouped_df).mark_bar().encode(
        x=alt.X(f"{x}:N", title=f"{x}"),
        y=alt.Y('Proportion:Q', title='Proportion', axis=alt.Axis(format='%')),
        color=alt.Color(f"{group}:N", scale=alt.Scale(range=color_list))
    ).properties(
        title=title
    )

    return chart

def facet_group_countplot(df: pd.DataFrame, x: str, group: str, facet: str, title: str) -> alt.Chart:
    '''
    Function to create a horizontal facet group count plot which groups by the group variable and facets by the facet variable.
    '''
    # Remove null values
    df = df[[x, group, facet]].dropna()

    # Create the base chart
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f"{x}:N", title=f"{x}"),
        y=alt.Y("count()", title="Count"),
        color=alt.Color(f"{group}:N", 
                       scale=alt.Scale(range=color_list),
                       title=group),
        column=alt.Column(f"{facet}:N", title=f"{facet}")
    ).properties(
        width=200,
        height=200,
        title=title
    )

    return chart

def sex_class_survival_countplot(df: pd.DataFrame, x: str, group: str, facet: str, title: str) -> alt.Chart:
    '''
    Function to create a horizontal facet group count plot showing gender distribution and survival rates.
    '''
    # Remove null values
    df = df[[x, group, facet]].dropna()

    # Create color scale for gender and survival combinations
    color_scale = alt.Scale(
        domain=['female_survived', 'female_died', 'male_survived', 'male_died'],
        range=['#FFB6C1', '#DC143C', '#A5D7E8', '#19376D']
    )

    # Create a new column combining gender and survival
    df['Gender_Survival'] = df.apply(
        lambda row: f"{'female' if row['Sex']=='female' else 'male'}_{'survived' if row['Survived']==1 else 'died'}", 
        axis=1
    )

    # Create the base chart
    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y(f"{x}:N", title=f"{x}", axis=alt.Axis(labelAngle=0)),
        x=alt.X("count():Q", title="Count", stack='zero'),
        color=alt.Color('Gender_Survival:N', scale=color_scale, 
                       legend=alt.Legend(title="Gender and Survival")),
        tooltip=[
            x,
            facet, 
            group,
            alt.Tooltip('count()', title='Count'),
            alt.Tooltip('Gender_Survival:N', title='Category')
        ]
    ).properties(
        width=600,
        height=200,
        title=title
    )

    return chart