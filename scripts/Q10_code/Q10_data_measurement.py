import pandas as pd
import os
import matplotlib.pyplot as plt
# Load the dataset
analysis_data_path = os.path.join('..', '..', 'data', 'Q10_analysis_data', 'Q10_analysis_data.csv')
pic_path = os.path.join('..', '..', 'data', 'Q10_analysis_data', 'Q10_data_plots')
pic_path_name = os.path.join(pic_path, 'Q10_data_summary_1.png')
df = pd.read_csv(analysis_data_path)
# Define a dictionary with cities as keys and lists of keywords as values
city_keywords = {
    'Dubai': ['Dubai'],
    'New York City': ['New York', 'NY', 'New York City'],
    'Paris': ['Paris'],
    'Rio de Janeiro': ['Rio', 'Rio de Janeiro']
}

# Initialize the list to hold the counts
city_data = []

# Loop through each city and its keywords
for city, keywords in city_keywords.items():
    # Initialize a counter for keyword occurrences
    keyword_occurrences = 0

    # Loop through each keyword
    for keyword in keywords:
        # Use a case-insensitive search for the keyword in 'Q10'
        keyword_occurrences += df['Q10'].str.contains(keyword, case=False, na=False).sum()

    # Calculate total responses for the city
    total_responses = df['Label'].value_counts()[city]

    # Calculate other mentions
    other_mentions = total_responses - keyword_occurrences

    # Calculate keyword occurrence percentage
    keyword_occurrence_percentage = (keyword_occurrences / total_responses) * 100

    # Append the data for this city to our list
    city_data.append({
        'City': city,
        'Keyword Occurrences': keyword_occurrences,
        'Other Mentions': other_mentions,
        'Total Responses': total_responses,
        'Keyword Occurrence (%)': f"{keyword_occurrence_percentage:.2f}%"
    })

# Create a DataFrame from the list
results_df = pd.DataFrame(city_data)

# Create a figure and a single subplot
fig, ax = plt.subplots(figsize=(12, 4))  # Adjust the size as needed

# Hide the axes
ax.axis('off')

# Add a table at the bottom of the axes and align it
the_table = ax.table(cellText=results_df.values,
                     colLabels=results_df.columns,
                     loc='center',
                     cellLoc='center',
                     colColours=["#CCCCCC"] * len(results_df.columns))  # Adding a grey color to the header

# Adjust the table scale and font sizethe header
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(1.2, 1.2)

# Save the figure
plt.savefig(pic_path_name, bbox_inches='tight', dpi=300)
plt.close()

# Loop through the city data and create pie charts
for city_info in city_data:
    # Data to plot
    labels = 'Keyword Occurrences', 'Other Mentions'
    sizes = [city_info['Keyword Occurrences'], city_info['Other Mentions']]
    colors = ['pink','yellow']  # You can choose your own colors
    explode = (0.1, 0)  # Only "explode" the 1st slice (i.e., 'Keyword Occurrences')

    # Plot
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    plt.title(f'Proportion of Keyword Occurrences for {city_info["City"]}')

    # Save the figure
    plt.savefig(os.path.join(pic_path, f"{city_info['City']}_keyword_proportion.png"), bbox_inches='tight')

    # Create a DataFrame for keywords
    keywords_summary = pd.DataFrame({
        'City': list(city_keywords.keys()),
        'Keywords': [', '.join(keywords) for keywords in city_keywords.values()]
    })

    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(8, 3))  # Adjust the size as needed

    # Hide the axes
    ax.axis('off')

    # Add a table at the bottom of the axes and align it
    the_table = ax.table(cellText=keywords_summary.values,
                         colLabels=keywords_summary.columns,
                         loc='center',
                         cellLoc='center',
                         colColours=["#CCCCCC"] * len(keywords_summary.columns))  # Color for the header

    # Adjust the table scale and font size
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1.2, 1.2)

    # Save the figure
    keywords_summary_image_path = os.path.join(pic_path, 'Q10_keywords_summary.png')
    plt.savefig(keywords_summary_image_path, bbox_inches='tight', dpi=300)

    # Close the plot to free up memory
    plt.close(fig)

