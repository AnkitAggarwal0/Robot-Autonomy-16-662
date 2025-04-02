import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files without headers
df1 = pd.read_csv('force_vs_time_fc.csv', header=None)
df2 = pd.read_csv('force_vs_time_ic.csv', header=None)


# Assuming the first column (index 0) is for x-axis and the second (index 1) for y-axis
# Create the plot
plt.figure(figsize=(10, 6))

# Plot the first line
plt.plot(df1[0], df1[1], label='Force Controller')

# Plot the second line
plt.plot(df2[0], df2[1], label='Impedance Controller')
plt.ylim(0, 25)
# Customize the plot
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('Force v/s Impedance Controller Graphs on Static Whiteboard')
plt.legend()

# Display the plot
plt.savefig('StaticWB.png', dpi=300, bbox_inches='tight')
plt.show()

df3 = pd.read_csv('force_vs_time_p2fc.csv', header=None)
df4 = pd.read_csv('force_vs_time_p2ic.csv', header=None)


# Assuming the first column (index 0) is for x-axis and the second (index 1) for y-axis
# Create the plot
plt.figure(figsize=(10, 6))

# Plot the first line
plt.plot(df3[0], df3[1], label='Force Controller')

# Plot the second line
plt.plot(df4[0], df4[1], label='Impedance Controller')

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Force')
plt.title('Force v/s Impedance Controller Graphs on Moving Whiteboard')

plt.ylim(0, 25)
plt.legend()

# Display the plot
plt.savefig('MovingWB.png', dpi=300, bbox_inches='tight')
plt.show()