
def new_plots(st, pd, np,  df, df_test):
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import gaussian_kde

    df_survived = df[df['Survived'] == 1]
    df_not_survived = df[df['Survived'] == 0]

    test_survived = df_test[df_test['Survived'] == 1]
    test_not_survived = df_test[df_test['Survived'] == 0]

    st.subheader('Correlation Matrix')
    col1, col2 = st.columns([1, 1])

    # Functie om een heatmap te maken met annotaties
    def create_heatmap(corr_matrix, title):
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[  # Wit naar donkerpaars kleurenschaal
                [0, 'white'],       # Wit voor lage waarden
                [1, '#800080']   # Donkerpaars voor hoge waarden
            ],
            colorbar=dict(title='Correlation'),
            zmin=-1, zmax=1,  # Zet de schaal van -1 tot 1
            text=np.round(corr_matrix.values, 2),  # Rond de waarden af op 2 decimalen
            hoverinfo='text',  # Zorg ervoor dat alleen tekst wordt getoond bij hover
            showscale=True
        ))

        # Voeg de annotaties toe (de correlatiewaarden in de vakjes)
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                fig.add_annotation(
                    text=str(np.round(corr_matrix.values[i, j], 2)),  # Voeg de correlatiewaarde toe
                    x=corr_matrix.columns[j],  # Kolom op de x-as
                    y=corr_matrix.columns[i],  # Rij op de y-as
                    showarrow=False,
                    font=dict(color="black")  # Tekstkleur zwart
                )

        # Layout instellingen
        fig.update_layout(
            xaxis_nticks=36,
            width=700, height=500,
            xaxis_title="Features",
            yaxis_title="Features",
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor the legend at the bottom
                y=1.1,  # Position it above the plot (1.1 means 10% above the top)
                xanchor="center",
                x=0.5  # Center the legend horizontally
            )
        )

        return fig

    with col1:
        # Verwijder de niet-numerieke kolommen en bereken de correlatiematrix voor de train dataset
        numeric_columns_train = df.select_dtypes(include=[np.number]).columns
        m_train = df[numeric_columns_train].corr()

        # Creëer de heatmap voor de train dataset
        fig_train = create_heatmap(m_train, 'Correlation Matrix')
        st.plotly_chart(fig_train)
    
    with col2:

        # Verwijder de niet-numerieke kolommen en bereken de correlatiematrix voor de test dataset
        numeric_columns_test = df_test.select_dtypes(include=[np.number]).columns
        m_test = df_test[numeric_columns_test].corr()

        # Creëer de heatmap voor de test dataset
        fig_test = create_heatmap(m_test, 'Correlation Matrix')
        st.plotly_chart(fig_test)
    
    st.write('---')
    st.subheader('Overlevingskansen')
    col1, col2 = st.columns([1, 1])

    def create_survival_pie(df):
            # Calculate survival counts
            survival_counts = df['Survived'].value_counts()
            
            # Create the pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Did Not Survive', 'Survived'],
                values=survival_counts.values,
                hole=0.4,  # Creates a donut chart
                marker_colors=['orchid', 'purple'],
                textinfo='label+percent',
                textposition='outside',
                pull=[0.1, 0],  # Pulls out the first slice slightly
                hovertemplate="<b>%{label}</b><br>" +
                            "Count: %{value}<br>" +
                            "Percentage: %{percent}<extra></extra>"
            )])

            # Update layout
            fig.update_layout(
                annotations=[{
                    'text': 'Survival<br>Rate',
                    'x': 0.5,
                    'y': 0.5,
                    'font_size': 16,
                    'showarrow': False
                }],
                showlegend=False,
                width=700,
                height=500
            )
            
            return fig
    
    with col1:
        # Create and display the chart
        fig = create_survival_pie(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create and display the chart
        fig = create_survival_pie(df_test)
        st.plotly_chart(fig, use_container_width=True)

    st.write('---')
    st.subheader('Ticket prijs tegen leeftijd met klasse')

    filter_survival = st.radio(
        'Filter by survival status:',
        ('Both', 'Survived', 'Not Survived'),
        horizontal=True
    )

    # Horizontal radio button to filter by class (Pclass)
    filter_class = st.radio(
        'Filter by class:',
        ('All', 'Class 1', 'Class 2', 'Class 3'),
        horizontal=True
    )


    col1, col2 = st.columns([1, 1])

            # Function to handle class filtering
    def class_filter(df, pclass):
        if filter_class != 'All':
            class_number = int(filter_class.split()[-1])  # Get the class number (1, 2, or 3)
            return df[df['Pclass'] == class_number]
        return df
    
    def create_scatter_plot(df_survived, df_not_survived):
        # Define symbol for each class
        symbols = {1: 'circle', 2: 'square', 3: 'diamond'}
        # Create a figure for the scatter plot
        fig = go.Figure()

        # Apply filtering for both survival status and class
        if filter_survival == 'Survived' or filter_survival == 'Both':
            df_survived_filtered = class_filter(df_survived, filter_class)
            for cls in [1, 2, 3]:
                if filter_class == 'All' or cls == int(filter_class.split()[-1]):
                    subset_survived = df_survived_filtered[df_survived_filtered['Pclass'] == cls]
                    fig.add_trace(go.Scatter(
                        x=subset_survived['Fare'],
                        y=subset_survived['Age'],
                        mode='markers',
                        name=f'Survived - Class {cls}',
                        marker=dict(
                            color='purple',
                            size=8,
                            symbol=symbols[cls],
                            opacity=0.7,
                            line=dict(width=1, color='black')
                        )
                    ))

        if filter_survival == 'Not Survived' or filter_survival == 'Both':
            df_not_survived_filtered = class_filter(df_not_survived, filter_class)
            for cls in [1, 2, 3]:
                if filter_class == 'All' or cls == int(filter_class.split()[-1]):
                    subset_not_survived = df_not_survived_filtered[df_not_survived_filtered['Pclass'] == cls]
                    fig.add_trace(go.Scatter(
                        x=subset_not_survived['Fare'],
                        y=subset_not_survived['Age'],
                        mode='markers',
                        name=f'Not Survived - Class {cls}',
                        marker=dict(
                            color='pink',
                            size=8,
                            symbol=symbols[cls],
                            opacity=0.7,
                            line=dict(width=1, color='black')
                        )
                    ))

        # Update layout settings
        fig.update_layout(
            xaxis_title='Ticket Prijs',
            yaxis_title='Leeftijd',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            )
        )

        return fig



    with col1:
        fig = create_scatter_plot(df_survived, df_not_survived)
        st.plotly_chart(fig)

    with col2:
        fig = create_scatter_plot(test_survived, test_not_survived)
        st.plotly_chart(fig)
    
    st.write('---')
    st.subheader('Ticket prijs verdeling per overlevings status')
    # make slider to adjust the fare range

    min_fare, max_fare = st.slider(
    'Selecteer Ticket Prijs Bereik',
    min_value=float(df['Fare'].min()),  # Minimum fare in the dataset
    max_value=float(df['Fare'].max()),  # Maximum fare in the dataset
    value=(float(df['Fare'].min()), float(df['Fare'].max()))  # Default range, up to 200
    )
    

    col1, col2 = st.columns([1, 1])

    def create_fare_boxplot(df):
        # Filter the dataframe based on the selected fare range
        df_filtered = df[(df['Fare'] >= min_fare) & (df['Fare'] <= max_fare)]

        # Create a new column to replace Survived with more descriptive labels
        df_filtered['Survived_Label'] = df_filtered['Survived'].replace({0: 'Not survived', 1: 'Survived'})

        # Create the boxplot for filtered data
        fig = px.box(
            df_filtered,
            x='Survived_Label',
            y='Fare',
            color='Survived_Label',
            color_discrete_map={'Not survived': 'pink', 'Survived': 'purple'}
        )

        # Update layout settings
        fig.update_layout(
            xaxis_title='Overlevings status',
            yaxis_title='Ticket Prijs',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            )
        )
        return fig
    
    with col1:
        fig = create_fare_boxplot(df)
        st.plotly_chart(fig)
    
    with col2:
        fig = create_fare_boxplot(df_test)
        st.plotly_chart(fig)
    
    st.write('---')
    st.subheader('Overlevingskansen per geslacht')
    col1, col2 = st.columns([1, 1])

    def create_gender_survival_pies(df):
        # Calculate survival rates for each gender
        male_stats = df[df['Sex'] == 'male']['Survived'].value_counts()
        female_stats = df[df['Sex'] == 'female']['Survived'].value_counts()
        
        # Create subplots for side-by-side pie charts
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "pie"}]],
            subplot_titles=('Male Passengers', 'Female Passengers')
        )
        
        # Add male survival pie chart
        fig.add_trace(
            go.Pie(
                labels=['Did Not Survive', 'Survived'],
                values=male_stats.values,
                hole=0.4,
                marker_colors=['orchid', 'purple'],
                textinfo='label+percent',
                textposition='outside',
                pull=[0.1, 0],
                hovertemplate="<b>%{label}</b><br>" +
                            "Count: %{value}<br>" +
                            "Percentage: %{percent}<extra></extra>",
                name="Male Passengers"
            ),
            row=1, col=1
        )
        
        # Add female survival pie chart
        fig.add_trace(
            go.Pie(
                labels=['Survived', 'Did Not Survive'],
                values=female_stats.values,
                hole=0.4,
                marker_colors=['purple', 'orchid'],
                textinfo='label+percent',
                textposition='outside',
                pull=[0.1, 0],
                hovertemplate="<b>%{label}</b><br>" +
                            "Count: %{value}<br>" +
                            "Percentage: %{percent}<extra></extra>",
                name="Female Passengers"
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            width=1000,
            height=500,
            annotations=[
                dict(text='Male<br>Survival Rate', x=0.20, y=0.5, font_size=14, showarrow=False),
                dict(text='Female<br>Survival Rate', x=0.80, y=0.5, font_size=14, showarrow=False)
            ]
        )
        
        return fig


    with col1:
        # Create and display the chart
        fig = create_gender_survival_pies(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Create and display the chart
        fig = create_gender_survival_pies(df_test)
        st.plotly_chart(fig, use_container_width=True)


    st.write('---')
    st.subheader('Overlevenden per vertrek locatie')
    col1, col2 = st.columns([1, 1])

    def create_embarked_bar_chart(df):
        # Groepeer de data op 'Embarked' en 'Survived', en tel de frequenties
        df_grouped = df.groupby(['Embarked', 'Survived']).size().reset_index(name='count')

        # Filter de data voor overlevenden en niet-overlevenden
        df_survived = df_grouped[df_grouped['Survived'] == 1]
        df_not_survived = df_grouped[df_grouped['Survived'] == 0]

        fig = go.Figure()

        # Voeg bar voor niet-overlevenden toe (roze)
        fig.add_trace(go.Bar(
            x=df_not_survived['Embarked'],
            y=df_not_survived['count'],
            name='Not Survived',
            marker_color='pink'
        ))

        # Voeg bar voor overlevenden toe (paars)
        fig.add_trace(go.Bar(
            x=df_survived['Embarked'],
            y=df_survived['count'],
            name='Survived',
            marker_color='purple'
        ))

        # Layout instellingen
        fig.update_layout(
            xaxis_title='Vertrek Locatie',
            yaxis_title='Aantal Mensen',
            barmode='group',
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor the legend at the bottom
                y=1.1,  # Position it above the plot (1.1 means 10% above the top)
                xanchor="center",
                x=0.5  # Center the legend horizontally
            )
        )

        return fig

    with col1:
        fig = create_embarked_bar_chart(df)
        st.plotly_chart(fig)

    with col2:
        fig = create_embarked_bar_chart(df_test)
        st.plotly_chart(fig)

    st.write('---')
    st.subheader('Leeftijd verdeling per passagiers klasse')
    col1, col2 = st.columns([1, 1])

    def create_histogram_class(df):
        # Filter de data op basis van klasse (Pclass)
        df_class_1 = df[df['Pclass'] == 1]
        df_class_2 = df[df['Pclass'] == 2]
        df_class_3 = df[df['Pclass'] == 3]

        fig = go.Figure()

        # Voeg histogram voor klasse 1 toe
        fig.add_trace(go.Histogram(
            x=df_class_1['Age'],
            name='Class 1',
            xbins=dict(size=5),  # Aantal bins voor de leeftijden
            marker_color='purple',
            opacity=0.5
        ))

        # Voeg histogram voor klasse 2 toe
        fig.add_trace(go.Histogram(
            x=df_class_2['Age'],
            name='Class 2',
            xbins=dict(size=5),
            marker_color='orchid',  # Kleur tussen roze en paars
            opacity=0.5
        ))

        # Voeg histogram voor klasse 3 toe
        fig.add_trace(go.Histogram(
            x=df_class_3['Age'],
            name='Class 3',
            xbins=dict(size=5),
            marker_color='pink',
            opacity=0.5
        ))

        # Layout instellingen
        fig.update_layout(
            xaxis_title='Leeftijd',
            yaxis_title='Aantal Passagiers',
            barmode='overlay',  # Zorgt dat de histogrammen overlappen
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor the legend at the bottom
                y=1.1,  # Position it above the plot (1.1 means 10% above the top)
                xanchor="center",
                x=0.5  # Center the legend horizontally
            )
        )
        return fig

    with col1:
        fig = create_histogram_class(df)
        st.plotly_chart(fig)
    
    with col2:
        fig = create_histogram_class(df_test)
        st.plotly_chart(fig)
    
    st.write('---')
    st.subheader('Leeftijd verdeling per passagiers klasse')
    col1, col2 = st.columns([1, 1])

    # Functie om KDE (vloeiende lijn) te berekenen
    def create_kde_line(data, label, color):
            kde = gaussian_kde(data.dropna())  # Drop NaN-waarden voor KDE
            x_vals = np.linspace(min(data.dropna()), max(data.dropna()), 200)  # X-waarden van min tot max leeftijd
            y_vals = kde(x_vals)  # Y-waarden van de KDE
            return go.Scatter(x=x_vals, y=y_vals, mode='lines', name=label, line=dict(color=color, width=3))
    
    def create_age_class_kde(df):
        # Filter de data op basis van klasse (Pclass)
        df_class_1 = df[df['Pclass'] == 1]['Age']
        df_class_2 = df[df['Pclass'] == 2]['Age']
        df_class_3 = df[df['Pclass'] == 3]['Age']

        fig = go.Figure()

        # Voeg KDE-lijn voor klasse 1 toe (paars)
        fig.add_trace(create_kde_line(df_class_1, 'Class 1', 'purple'))

        # Voeg KDE-lijn voor klasse 2 toe (magenta)
        fig.add_trace(create_kde_line(df_class_2, 'Class 2', 'orchid'))

        # Voeg KDE-lijn voor klasse 3 toe (roze)
        fig.add_trace(create_kde_line(df_class_3, 'Class 3', 'pink'))

        # Layout instellingen
        fig.update_layout(
            xaxis_title='Leeftijd',
            yaxis_title='Dichtheid',
            xaxis=dict(range=[min(df['Age']), max(df['Age'])]),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor the legend at the bottom
                y=1.1,  # Position it above the plot (1.1 means 10% above the top)
                xanchor="center",
                x=0.5  # Center the legend horizontally
            )
        )
        return fig

    with col1:
        fig = create_age_class_kde(df)
        st.plotly_chart(fig)
    
    with col2:
        fig = create_age_class_kde(df_test)
        st.plotly_chart(fig)
        
    st.write('---')
    st.subheader('Overlevingsaantal en -percentage per Dek')
    col1, col2 = st.columns([1, 1])

    def deck_survival(df):
        # Calculate the counts of survived and not survived passengers per deck
        survived_counts = df[df['Survived'] == 1].groupby('Deck').size()
        not_survived_counts = df[df['Survived'] == 0].groupby('Deck').size()

        # Calculate total passengers per deck
        total_counts = survived_counts + not_survived_counts

        # Calculate survival rate per deck (number of survivors / total number of passengers)
        survival_rate = survived_counts / total_counts

        # Create a bar trace for survived passengers
        trace_survived = go.Bar(
            x=survived_counts.index,  # Deck
            y=survived_counts.values,  # Number of survivors
            name='Survived',
            marker=dict(color='purple')  # Purple for survivors
        )

        # Create a bar trace for non-survived passengers
        trace_not_survived = go.Bar(
            x=not_survived_counts.index,  # Deck
            y=not_survived_counts.values,  # Number of non-survivors
            name='Not Survived',
            marker=dict(color='pink')  # Pink for non-survivors
        )

        # Create a line trace for survival rate
        trace_survival_rate = go.Scatter(
            x=survival_rate.index,  # Deck
            y=survival_rate.values,  # Survival rate
            name='Survival Rate',
            mode='lines+markers',
            yaxis='y2',  # Plot on secondary y-axis
            line=dict(color='orchid', width=3),  # Blue line
            marker=dict(size=8)  # Marker size for the points
        )

        # Create the figure and add all traces
        fig = go.Figure(data=[trace_survived, trace_not_survived, trace_survival_rate])

        # Update layout settings, including a secondary y-axis for the survival rate
        fig.update_layout(
            xaxis_title='Dek',
            yaxis_title='Aantal Passagiers',
            barmode='group',  # Group the bars side by side
            yaxis2=dict(
                overlaying='y',  # Overlay on top of the primary y-axis
                side='right',
                range=[0, 1]  # Survival rate is a proportion between 0 and 1
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor the legend at the bottom
                y=1.1,  # Position it above the plot (1.1 means 10% above the top)
                xanchor="center",
                x=0.5  # Center the legend horizontally
            )
        )
        return fig

    with col1:
        fig = deck_survival(df)
        st.plotly_chart(fig)

    with col2:
        fig = deck_survival(df_test)
        st.plotly_chart(fig)


    st.write('---')
    st.subheader('Overlevingskansen per titel')

    col1, col2 = st.columns([1, 1])

    def survival_per_title(df):
        # Group the data by Title and Survived status
        survived_counts = df[df['Survived'] == 1].groupby('Title').size()
        not_survived_counts = df[df['Survived'] == 0].groupby('Title').size()

        # Create a bar trace for survived passengers
        trace_survived = go.Bar(
            x=survived_counts.index,  # Titles
            y=survived_counts.values,  # Number of survivors
            name='Survived',
            marker=dict(color='purple')  # Purple for survivors
        )

        # Create a bar trace for non-survived passengers
        trace_not_survived = go.Bar(
            x=not_survived_counts.index,  # Titles
            y=not_survived_counts.values,  # Number of non-survivors
            name='Not Survived',
            marker=dict(color='pink')  # Pink for non-survivors
        )

        # Calculate survival chances
        total_counts = survived_counts + not_survived_counts
        survival_chance = survived_counts / total_counts

        # Create a line trace for survival chance
        trace_survival_chance = go.Scatter(
            x=survival_chance.index,  # Titles
            y=survival_chance.values,  # Survival chances
            name='Survival Chance',
            mode='lines+markers',  # Show both line and markers
            yaxis='y2',  # Plot on secondary y-axis
            line=dict(color='orchid', width=3),  # Blue line
            marker=dict(size=8)  # Marker size for the points
        )

        # Create the figure and add both traces
        fig = go.Figure(data=[trace_survived, trace_not_survived, trace_survival_chance])

        # Update layout settings
        fig.update_layout(
            xaxis_title='Aanspreektitel',
            yaxis_title='Aantal Passagiers',
            barmode='group',  # Group the bars side by side
            yaxis2=dict(
                overlaying='y',  # Overlay on top of the primary y-axis
                side='right',
                range=[0, 1]  # Survival rate is a proportion between 0 and 1
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor the legend at the bottom
                y=1.1,  # Position it above the plot
                xanchor="center",
                x=0.5  # Center the legend horizontally
            )
        )
        return fig

    with col1:
        fig = survival_per_title(df)
        st.plotly_chart(fig)

    with col2:
        fig = survival_per_title(df_test)
        st.plotly_chart(fig)
    
    st.write('---')
    st.subheader('Aantal VIPs per Dek')
    col1, col2 = st.columns([1, 1])

    def vip_per_deck(df):
       # Filter the DataFrame for passengers with the title 'VIP'
        vip_data = df[df['Title'] == 'Vip']

        # Group by Deck and count the number of VIPs per deck
        vip_counts = vip_data['Deck'].value_counts()

        # Create a bar trace for VIP counts per deck
        trace_vip = go.Bar(
            x=vip_counts.index,  # Decks
            y=vip_counts.values,  # Number of VIPs
            name='VIP Count',
            marker=dict(color='purple')
        )

        # Create the figure and add the trace
        fig = go.Figure(data=[trace_vip])

        # Update layout settings
        fig.update_layout(
            xaxis_title='Deck',
            yaxis_title='Aantal VIPs',
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor the legend at the bottom
                y=1.1,  # Position it above the plot
                xanchor="center",
                x=0.5  # Center the legend horizontally
            )
        )
        return fig

    with col1:
        fig = vip_per_deck(df)
        st.plotly_chart(fig)

    with col2:
        fig = vip_per_deck(df_test)
        st.plotly_chart(fig)