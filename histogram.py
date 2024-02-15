import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.size'] = 24

# Data
age_groups = ['10-19', '20-29', '30-39', '40-49', '50-59']
males_counts = [17, 66, 0, 0, 0]  # Number of males for each age group
females_counts = [14, 35, 2, 0, 0]  # Number of females for each age group

bar_width = 0.4  # Width of the bars
index = np.arange(len(age_groups))  # X-axis indices
males_counts = list(map(lambda x: x * 5, males_counts))
females_counts = list(map(lambda x: x * 5, females_counts))

plt.figure(figsize=(10, 6))
print(males_counts)
# Create a black and white (grayscale) plot
plt.bar(index - bar_width/2, males_counts, bar_width, label='Males', color='gray', edgecolor='black', hatch='/', lw=2)
plt.bar(index + bar_width/2, females_counts, bar_width, label='Females', color='lightgray', edgecolor='black', hatch='\\', lw=2)

# Change to the desired font size
plt.xlabel('Age Group')
plt.ylabel('Count')

# Set X-axis ticks and labels for the entire age range
plt.xticks(index, age_groups, rotation=45, ha='right')  # Rotate labels for better readability

#plt.tight_layout()
# Save the figure with higher DPI (e.g., 300)
#plt.savefig("fhd_plot.png", dpi=300)
#
#

# Show the graph
plt.legend()
plt.tight_layout()  # Adjust layout for better spacing
plt.show()

