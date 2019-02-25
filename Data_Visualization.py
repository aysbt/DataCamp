#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import matplotlib.pyplot as plt
u = np.linspace(-2,2,3)
v = np.linspace(-1,1,5)
X,Y = np.meshgrid(u,v)
Z = X**2/25 + Y**2/4
print(Z)
plt.set_cmap('Pastel1')
plt.pcolor(Z) 
plt.show()


# In[42]:


P = np.array([[1,2,3],[4,5,6]])
print(P)
plt.set_cmap('Greys')
plt.pcolor(P)
plt.show()


# # Generating meshes
# In order to visualize two-dimensional arrays of data, it is necessary to understand how to generate and manipulate 2-D arrays. Many Matplotlib plots support arrays as input and in particular, they support NumPy arrays. The NumPy library is the most widely-supported means for supporting numeric arrays in Python.
# 
# In this exercise, you will use the meshgrid function in NumPy to generate 2-D arrays which you will then visualize using plt.imshow(). The simplest way to generate a meshgrid is as follows:
# 
# import numpy as np
# Y,X = np.meshgrid(range(10),range(20))
# This will create two arrays with a shape of (20,10), which corresponds to 20 rows along the Y-axis and 10 columns along the X-axis. In this exercise, you will use np.meshgrid() to generate a regular 2-D sampling of a mathematical function.
# 
# 1-Import the numpy and matplotlib.pyplot modules using the respective aliases np and plt.
# 2-Generate two one-dimensional arrays u and v using np.linspace(). The array u should contain 41 values uniformly spaced beween -2 and +2. The array v should contain 21 values uniformly spaced between -1 and +1.
# Construct two two-dimensional arrays X and Y from u and v using np.meshgrid(). The resulting arrays should have shape (41,21).
# 3-After the array Z is computed using X and Y, visualize the array Z using plt.pcolor() and plt.show().
# Z = np.sin(3*np.sqrt(X**2 + Y**2))
# 4-Save the resulting figure as 'sine_mesh.png'

# In[43]:



import numpy as np
import matplotlib.pyplot as plt

u = np.linspace(-2,2,41)
v = np.linspace(-1,1,21)
X,Y = np.meshgrid(u,v)
Z = np.sin(3*np.sqrt(X**2 + Y**2))
plt.set_cmap('CMRmap')
plt.pcolor(Z)
plt.show()


# In[44]:


A = np.array([[1, 0, -1], [2, 0, 1], [1, 1, 1]])
plt.pcolor(A, cmap='Blues')
plt.show()


# In[45]:


A = np.array([[1, 1, 1], [2, 0, 1], [1, 0, -1]])
plt.pcolor(A, cmap='Blues')
plt.show()


# # Bivariate function
# 

# In[46]:


r = np.linspace(-2,2,63)
d = np.linspace(-1,1,33)
X,Y =np.meshgrid(r,d)
Z = X**2/25 + Y**2/4
plt.set_cmap('Greys')
plt.pcolor(X,Y,Z)
plt.colorbar()
plt.show()


# In[47]:


plt.contour(X,Y,Z,30, cmap ='CMRmap_r')
plt.show()


# 1-Using the meshgrid X, Y as axes:
# 2-Generate a default contour plot of the array Z in the upper left subplot.
# 3-Generate a contour plot of the array Z in the upper right subplot with 20 contours.
# 4-Generate a default filled contour plot of the array Z in the lower left subplot.
# 5-Generate a default filled contour plot of the array Z in the lower right subplot with 20 contours.
# 6-Improve the spacing between the subplots with plt.tight_layout() and display the figure.

# In[48]:


plt.set_cmap('CMRmap')
plt.subplot(2,2,1)
plt.contour(X,Y,Z)
plt.subplot(2,2,2)
plt.contour(X,Y,Z,20)
plt.subplot(2,2,3)
plt.contourf(X,Y,Z)
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20)
plt.tight_layout()
plt.show()


# # Modifying colormaps
# When displaying a 2-D array with plt.imshow() or plt.pcolor(), the values of the array are mapped to a corresponding color. The set of colors used is determined by a colormap which smoothly maps values to colors, making it easy to understand the structure of the data at a glance.
# 
# It is often useful to change the colormap from the default 'jet' colormap used by matplotlib. A good colormap is visually pleasing and conveys the structure of the data faithfully and in a way that makes sense for the application.
# 
# Some matplotlib colormaps have unique names such as 'jet', 'coolwarm', 'magma' and 'viridis'.
# Others have a naming scheme based on overall color such as 'Greens', 'Blues', 'Reds', and 'Purples'.
# Another four colormaps are based on the seasons, namely 'summer', 'autumn', 'winter' and 'spring'.
# You can insert the option cmap=<name> into most matplotlib functions to change the color map of the resulting plot.
# In this exercise, you will explore four different colormaps together using plt.subplot(). You will use a pregenerated array Z and a meshgrid X, Y to generate the same filled contour plot with four different color maps. Be sure to also add a color bar to each filled contour plot with plt.colorbar().

# In[42]:


plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar
plt.title('Gray')
plt.subplot(2,2,3)
plt.contourf(X,Y,X,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20, cmap= 'winter')
plt.colorbar()
plt.title('Winter')
plt.tight_layout()
plt.show()


# # Using hist2d()
# Given a set of ordered pairs describing data points, you can count the number of points with similar values to construct a two-dimensional histogram. This is similar to a one-dimensional histogram, but it describes the joint variation of two random variables rather than just one.
# 
# In matplotlib, one function to visualize 2-D histograms is plt.hist2d().
# 
# You specify the coordinates of the points using plt.hist2d(x,y) assuming x and y are two vectors of the same length.
# You can specify the number of bins with the argument bins=(nx, ny) where nx is the number of bins to use in the horizontal direction and ny is the number of bins to use in the vertical direction.
# You can specify the rectangular region in which the samples are counted in constructing the 2D histogram. The optional parameter required is range=((xmin, xmax), (ymin, ymax)) where
# xmin and xmax are the respective lower and upper limits for the variables on the x-axis and
# ymin and ymax are the respective lower and upper limits for the variables on the y-axis. Notice that the optional range argument can use nested tuples or lists.
# In this exercise, you'll use some data from the auto-mpg data set. There are two arrays mpg and hp that respectively contain miles per gallon and horse power ratings from over three hundred automobiles built.
# 1- first read the auto-mpg.csv file from inside the data file
# 2-Generate a two-dimensional histogram to view the joint variation of the mpg and hp arrays.
# 3-Put hp along the horizontal axis and mpg along the vertical axis.
# 4-Specify 20 by 20 rectangular bins with the bins argument.
# 5-Specify the region covered with the optional range argument so that the plot samples hp between 40 and 235 on the x-axis and mpg between 8 and 48 on the y-axis.
# 6-Add a color bar to the histogram.
# 

# In[64]:


import pandas as pd
import matplotlib.pyplot as plt
auto = pd.read_csv('data/auto-mpg.csv')
mpg = auto['mpg']
mpg =np.array(mpg)
#print(mpg)
hp =auto['hp']
hp = np.array(hp)
#print(hp)
plt.hist2d(hp, mpg, bins= (20,20) , range=((40,235),(8,48)) )
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()


# # Using hexbin()
# The function plt.hist2d() uses rectangular bins to construct a two dimensional histogram. As an alternative, the function plt.hexbin() uses hexagonal bins. The underlying algorithm (based on this article from 1987) constructs a hexagonal tesselation of a planar region and aggregates points inside hexagonal bins.
# 
# The optional gridsize argument (default 100) gives the number of hexagons across the x-direction used in the hexagonal tiling. If specified as a list or a tuple of length two, gridsize fixes the number of hexagon in the x- and y-directions respectively in the tiling.
# The optional parameter extent=(xmin, xmax, ymin, ymax) specifies rectangular region covered by the hexagonal tiling. In that case, xmin and xmax are the respective lower and upper limits for the variables on the x-axis and ymin and ymax are the respective lower and upper limits for the variables on the y-axis.
# In this exercise, you'll use the same auto-mpg data as in the last exercise (again using arrays mpg and hp). This time, you'll use plt.hexbin() to visualize the two-dimensional histogram.
# 
# Instructions
# 
# 1-Generate a two-dimensional histogram with plt.hexbin() to view the joint variation of the mpg and hp vectors.
# 2-Put hp along the horizontal axis and mpg along the vertical axis.
# 3-Specify a hexagonal tesselation with 15 hexagons across the x-direction and 12 hexagons across the y-direction using gridsize.
# 4-Specify the rectangular region covered with the optional extent argument: use hp from 40 to 235 and mpg from 8 to 48.
# 5-Add a color bar to the histogram.

# In[66]:


plt.hexbin(hp,mpg, gridsize= (15,12), extent= (40,235,8,48))
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()


# # Loading, examining images
# Color images such as photographs contain the intensity of the red, green and blue color channels.
# 
# To read an image from file, use plt.imread() by passing the path to a file, such as a PNG or JPG file.
# The color image can be plotted as usual using plt.imshow().
# The resulting image loaded is a NumPy array of three dimensions. The array typically has dimensions M×N×3, where M×N is the dimensions of the image. The third dimensions are referred to as color channels (typically red, green, and blue).
# The color channels can be extracted by Numpy array slicing.
# 
# 1-Load the file 'Astronaut.jpg' into an array. (the image inside the img file)
# Print the shape of the img array. How wide and tall do you expect the image to be?
# Prepare img for display using plt.imshow().
# Turn off the axes using plt.axis('off').

# In[69]:


img = plt.imread('img/Astronaut.jpeg')
print(img.shape)
plt.imshow(img)
plt.axis('off')
plt.show()


# # Pseudocolor plot from image data
# Image data comes in many forms and it is not always appropriate to display the available channels in RGB space. In many situations, an image may be processed and analysed in some way before it is visualized in pseudocolor, also known as 'false' color.
# 
# In this exercise, you will perform a simple analysis using the image showing an astronaut as viewed from space. Instead of simply displaying the image, you will compute the total intensity across the red, green and blue channels. The result is a single two dimensional array which you will display using plt.imshow() with the 'gray' colormap.
# 
# 1-Compute the sum of the red, green, and blue channels of img by using the .sum() method with axis=2.
# 2-Print the shape of the intensity array to verify this is the shape you expect.
# 3-Plot intensity with plt.imshow() using a 'gray' colormap.
# 4-Add a colorbar to the figure.

# In[71]:


intensity = img.sum(axis=2)
print(intensity.shape)
plt.imshow(intensity, cmap='gray')
plt.colorbar()
plt.axis('off')
plt.show()


# # Extent and aspect
# When using plt.imshow() to display an array, the default behavior is to keep pixels square so that the height to width ratio of the output matches the ratio determined by the shape of the array. In addition, by default, the x- and y-axes are labeled by the number of samples in each direction.
# 
# The ratio of the displayed width to height is known as the image aspect and the range used to label the x- and y-axes is known as the image extent. The default aspect value of 'auto' keeps the pixels square and the extents are automatically computed from the shape of the array if not specified otherwise.
# 
# In this exercise, you will investigate how to set these options explicitly by plotting the same image in a 2 by 2 grid of subplots with distinct aspect and extent options.
# 
# Instructions
# 
# 1-Display img in the top left subplot with horizontal extent from -1 to 1, vertical extent from -1 to 1, and aspect ratio 0.5.
# 2-Display img in the top right subplot with horizontal extent from -1 to 1, vertical extent from -1 to 1, and aspect ratio 1.
# 3-Display img in the bottom left subplot with horizontal extent from -1 to 1, vertical extent from -1 to 1, and aspect ratio 2.
# 4-Display img in the bottom right subplot with horizontal extent from -2 to 2, vertical extent from -1 to 1, and aspect ratio 2.

# In[77]:


plt.subplot(2,2,1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5') 
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=0.5)

plt.subplot(2,2,2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1,0,1])
plt.yticks([-1,0-1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=1)

plt.subplot(2,2,3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0-1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=2)

plt.subplot(2,2,4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0-1])
plt.imshow(img, extent=(-2,2,-1,1), aspect=2)

plt.tight_layout()
plt.show()


# # Rescaling pixel intensities
# Sometimes, low contrast images can be improved by rescaling their intensities. For instance, this image of Hawkes Bay, New Zealand (originally by Phillip Capper, modified by User:Konstable, via Wikimedia Commons, CC BY 2.0) has no pixel values near 0 or near 255 (the limits of valid intensities).
# 
# For this exercise, you will do a simple rescaling (remember, an image is NumPy array) to translate and stretch the pixel intensities so that the intensities of the new image fill the range from 0 to 255.
# 
# Instructions
# 2- Load 'img/800px-Unequalized_Hawkes_Bay_NZ.jpg' using the plt.imread().
# 3-Use the methods .min() and .max() to save the minimum and maximum values from the array image as pmin and pmax respectively.
# 3-Create a new 2-D array rescaled_image using 256*(image-pmin)/(pmax-pmin)
# 4-Plot the original array image in the top subplot of a 2×1 grid.
# 5-Plot the new array rescaled_image in the bottom subplot of a 2×1 grid.

# In[84]:


image = plt.imread('img/800px-Unequalized_Hawkes_Bay_NZ.jpg')
pmin, pmax = image.min(), image.max()
print("The smallest &largest pixel intensities are %d & %d" %(pmin, pmax))
rescaled_image = 256*(image-pmin)/(pmax-pmin)
print("The rescaled smallest &largest pixel intensities are %.1f & %.1f" %(rescaled_image.min(), rescaled_image.max()))

plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image)

plt.subplot(2,1,2)
plt.title('rescaled image')
plt.axis('off')
plt.imshow(rescaled_image)

plt.show()


# # SEABORN (Statistical Data Visualization )
# 
# One of the simplest things you can do using seaborn is to fit and visualize a simple linear regression between two variables using sns.lmplot().
# 
# One difference between seaborn and regular matplotlib plotting is that you can pass pandas DataFrames directly to the plot and refer to each column by name. For example, if you were to plot the column 'price' vs the column 'area' from a DataFrame df, you could call sns.lmplot(x='area', y='price', data=df)
# 
# In this exercise, you will once again use the DataFrame auto containing the auto-mpg dataset. You will plot a linear regression illustrating the relationship between automobile weight and horse power.
# 1-Import matplotlib.pyplot and seaborn using the standard names plt and sns respectively.
# 2-Plot a linear regression between the 'weight' column (on the x-axis) and the 'hp' column (on the y-axis) from the DataFrame auto.
# 3-Display the plot as usual with plt.show(). This has been done for you, so hit 'Submit Answer' to view the plot.

# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
auto = pd.read_csv('data/auto-mpg.csv')
#print(auto.columns)
sns.lmplot(x='weight', y='hp', data=auto)
plt.show()


# Often, you don't just want to see the regression itself but also see the residuals to get a better idea how well the regression captured the data. Seaborn provides sns.residplot() for that purpose, visualizing how far datapoints diverge from the regression line.
# 
# In this exercise, you will visualize the residuals of a regression between the 'hp' column (horse power) and the 'mpg' column (miles per gallon) of the auto DataFrame used previously.
# 
# Instructions
# 
# 1-Import matplotlib.pyplot and seaborn using the standard names plt and sns respectively.
# import matplotlib.pyplot as plt
# import seaborn as sns
# 2-Generate a green residual plot of the regression between 'hp' (on the x-axis) and 'mpg' (on the y-axis). You will need to specify the additional data and color parameters.
# 3-Display the plot as usual using plt.show(). This has been done for you, so hit 'Submit Answer' to view the plot.

# In[9]:


sns.residplot(x='hp', y='mpg', data=auto, color='green')
plt.show()


# # Higher-order regressions
# When there are more complex relationships between two variables, a simple first order regression is often not sufficient to accurately capture the relationship between the variables. Seaborn makes it simple to compute and visualize regressions of varying orders.
# 
# Here, you will plot a second order regression between the horse power ('hp') and miles per gallon ('mpg') using sns.regplot() (the function sns.lmplot() is a higher-level interface to sns.regplot()). However, before plotting this relationship, compare how the residual changes depending on the order of the regression. Does a second order regression perform significantly better than a simple linear regression?
# 
# A principal difference between sns.lmplot() and sns.regplot() is the way in which matplotlib options are passed (sns.regplot() is more permissive).
# For both sns.lmplot() and sns.regplot(), the keyword order is used to control the order of polynomial regression.
# The function sns.regplot() uses the argument scatter=None to prevent plotting the scatter plot points again.
# Instructions
# 
# 1-Draw to plt.scatter() to ploton  auto['weight'] on the x-axis and auto['mpg'] on the y-axis, filled with red circles and with label='data'.
# 2-Plot a linear regression line of 'order 1' between 'weight' and 'mpg' in 'blue' without the scatter points.
# You need to specify the label and color parameters, in addition to scatter=None.
# 3-Plot a linear regression line of 'order 2' between 'weight' and 'mpg' in 'green' without the scatter points.
# -To force a higher order regression, you need to specify the order parameter. Here, it should be 2.
# 4-Add a legend to the 'upper right'.

# In[10]:


plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='blue', label='order 1')
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, order=2, color='green', label='order 2')

plt.legend(loc='upper right')
plt.show()


# # Grouping linear regressions by hue
# Often it is useful to compare and contrast trends between different groups. Seaborn makes it possible to apply linear regressions separately for subsets of the data by applying a groupby operation. Using the hue argument, you can specify a categorical variable by which to group data observations. The distinct groups of points are used to produce distinct regressions with different hues in the plot.
# 
# In the automobile dataset - which has been pre-loaded here as auto - you can view the relationship between weight ('weight') and horsepower ('hp') of the cars and group them by their origin ('origin'), giving you a quick visual indication how the relationship differs by continent.
# 
# Instructions
# 
# 1-Plot a linear regression between 'weight' and 'hp' grouped by 'origin'.
# 2-Use the keyword argument hue to group rows with the categorical column 'origin'.
# 3-Use the keyword argument palette to specify the 'Set1' palette for coloring the distinct groups.

# In[14]:


sns.lmplot(x='weight', y='hp', data=auto, hue='origin', palette='Set1')
plt.show()


# # Grouping linear regressions by row or column
# Rather than overlaying linear regressions of grouped data in the same plot, we may want to use a grid of subplots. The sns.lmplot() accepts the arguments row and/or col to arrangements of subplots for regressions.
# 1-Plot linear regressions of 'hp' (on the y-axis) versus 'weight' (on the x-axis) grouped row-wise by 'origin' from DataFrame auto.
# 2-Use the keyword argument row to group observations with the categorical column 'origin' in subplots organized in rows.

# In[19]:


sns.lmplot(x='weight', y='hp', data=auto, hue='origin', row='origin')
plt.show()


# # Constructing strip plots
# Regressions are useful to understand relationships between two continuous variables. Often we want to explore how the distribution of a single continuous variable is affected by a second categorical variable. Seaborn provides a variety of plot types to perform these types of comparisons between univariate distributions.
# 
# The strip plot is one way of visualizing this kind of data. It plots the distribution of variables for each category as individual datapoints. For vertical strip plots (the default), distributions of continuous values are laid out parallel to the y-axis and the distinct categories are spaced out along the x-axis.
# 
# For example, sns.stripplot(x='type', y='length', data=df) produces a sequence of vertical strip plots of length distributions grouped by type (assuming length is a continuous column and type is a categorical column of the DataFrame df).
# Overlapping points can be difficult to distinguish in strip plots. The argument jitter=True helps spread out overlapping points.
# Other matplotlib arguments can be passed to sns.stripplot(), e.g., marker, color, size, etc.
# Instructions
# 
# 1-In the first row of subplots, make a strip plot showing distribution of 'hp' values grouped horizontally by 'cyl'.
# 2-In the second row of subplots, make a second strip plot with improved readability. In particular, you'll call sns.stripplot() again, this time adding jitter=True and decreasing the point size to 3 using the size parameter.

# In[20]:


plt.subplot(2,1,1)
sns.stripplot(x='cyl', y='hp', data=auto)

plt.subplot(2,1,2)
sns.stripplot(x='cyl', y='hp', data=auto, size=3, jitter=True)

plt.show()


# # Constructing swarm plots
# As you have seen, a strip plot can be visually crowded even with jitter applied and smaller point sizes. An alternative is provided by the swarm plot (sns.swarmplot()), which is very similar but spreads out the points to avoid overlap and provides a better visual overview of the data.
# 
# The syntax for sns.swarmplot() is similar to that of sns.stripplot(), e.g., sns.swarmplot(x='type', y='length', data=df).
# The orientation for the continuous variable in the strip/swarm plot can be inferred from the choice of the columns x and y from the DataFrame data. The orientation can be set explicitly using orient='h' (horizontal) or orient='v' (vertical).
# Another grouping can be added in using the hue keyword. For instance, using sns.swarmplot(x='type', y='length', data=df, hue='build year') makes a swarm plot from the DataFrame df with the 'length' column values spread out vertically, horizontally grouped by the column 'type' and each point colored by the categorical column 'build year'.
# In this exercise, you'll use the auto DataFrame again to illustrate the use of sns.swarmplot() with grouping by hue and with explicit specification of the orientation using the keyword orient.
# 
# Instructions
# 100 XP
# In the first row of subplots, make a swarm plot showing distribution of 'hp' values grouped horizontally by 'cyl'.
# In the second row of subplots, make a second swarm plot with horizontal orientation (i.e., grouped vertically by 'cyl' with 'hp' value spread out horizontally) with points colored by 'origin'. You need to specify the orient parameter to explicitly set the horizontal orientation.

# In[23]:


plt.subplot(2,1,1)
sns.swarmplot(x='cyl', y='hp', data=auto)

plt.subplot(2,1,2)
sns.swarmplot(x='hp', y='cyl', data=auto, orient='h', hue='origin')

plt.show()


# # Constructing violin plots
# Both strip and swarm plots visualize all the datapoints. For large datasets, this can result in significant overplotting. Therefore, it is often useful to use plot types which reduce a dataset to more descriptive statistics and provide a good summary of the data. Box and whisker plots are a classic way of summarizing univariate distributions but seaborn provides a more sophisticated extension of the standard box plot, called a violin plot.
# 
# Here, you will produce violin plots of the distribution of horse power ('hp') by the number of cylinders ('cyl'). Additionally, you will combine two different plot types by overlaying a strip plot on the violin plot.
# 
# As before, the DataFrame has been pre-loaded for you as auto.
# 
# Instructions
# 100 XP
# In the first row of subplots, make a violin plot showing the distribution of 'hp' grouped by 'cyl'.
# In the second row of subplots, make a second violin plot without the inner annotations (by specifying inner=None) and with the color 'lightgray'.
# In the second row of subplots, overlay a strip plot with jitter and a point size of 1.5.

# In[26]:


plt.subplot(2,1,1)
sns.violinplot(x='cyl', y='hp', data=auto)

plt.subplot(2,1,2)
sns.violinplot(x='cyl', y='hp', data=auto, inner=None, color='lightgray')
sns.stripplot(x='cyl', y='hp', data=auto, size=1.5, jitter=True)
plt.tight_layout()
plt.show()


# # Plotting joint distributions (1)
# There are numerous strategies to visualize how pairs of continuous random variables vary jointly. Regression and residual plots are one strategy. Another is to visualize a bivariate distribution.
# 
# Seaborn's sns.jointplot() provides means of visualizing bivariate distributions. The basic calling syntax is similar to that of sns.lmplot(). By default, calling sns.jointplot(x, y, data) renders a few things:
# 
# A scatter plot using the specified columns x and y from the DataFrame data.
# A (univariate) histogram along the top of the scatter plot showing distribution of the column x.
# A (univariate) histogram along the right of the scatter plot showing distribution of the column y.
# 
# Instructions
# Use sns.jointplot() to visualize the joint variation of the columns 'hp' (on the x-axis) and 'mpg' (on the y-axis) from the DataFrame auto.
# 

# In[27]:


sns.jointplot(x='hp', y='mpg', data=auto)
plt.show()


# # Plotting joint distributions (2)
# The seaborn function sns.jointplot() has a parameter kind to specify how to visualize the joint variation of two continuous random variables (i.e., two columns of a DataFrame)
# 
# ##### kind='scatter' uses a scatter plot of the data points
# ##### kind='reg' uses a regression plot (default order 1)
# ##### kind='resid' uses a residual plot
# ##### kind='kde' uses a kernel density estimate of the joint distribution
# ##### kind='hex' uses a hexbin plot of the joint distribution
# For this exercise, you will again use sns.jointplot() to display the joint distribution of the hp and mpg columns of the auto DataFrame. This time, you will use kind='hex' to generate a hexbin plot of the joint distribution.
# 
# Instructions
# Create a hexbin plot of the joint distribution between 'hp' and 'mpg'.

# In[28]:


sns.jointplot(x='hp', y='mpg', data=auto, kind='hex')
plt.show()


# # Plotting distributions pairwise (1)
# Data sets often contain more than two continuous variables. The function sns.jointplot() is restricted to representing joint variation between only two quantities (i.e., two columns of a DataFrame). Visualizing multivariate relationships is trickier.
# 
# The function sns.pairplot() constructs a grid of all joint plots pairwise from all pairs of (non-categorical) columns in a DataFrame. The syntax is very simple: sns.pairplot(df), where df is a DataFrame. The non-categorical columns are identified and the corresponding joint plots are plotted in a square grid of subplots. The diagonal of the subplot grid shows the univariate histograms of the individual columns.
# 
# In this exercise, you will use a DataFrame auto comprising only three columns from the original auto-mpg data set.
# 
# Instructions
# 1-Plot the joint distributions between columns from the entire DataFrame auto.

# In[33]:


sns.pairplot(auto)
plt.show()


# # Plotting distributions pairwise (2)
# In this exercise, you will generate pairwise joint distributions again. This time, you will make two particular additions:
# 
# You will display regressions as well as scatter plots in the off-diagonal subplots. You will do this with the argument kind='reg' (where 'reg' means 'regression'). Another option for kind is 'scatter' (the default) that plots scatter plots in the off-diagonal subplots.
# You will also visualize the joint distributions separated by continent of origin. You will do this with the keyword argument hue specifying the 'origin'.
# Instructions
# 1-Plot the pairwise joint distributions separated by continent of origin and display the regressions.

# In[35]:


sns.pairplot(auto, hue='origin', kind='reg')
plt.show()


# # Visualizing correlations with a heatmap
# Plotting relationships between many variables using a pair plot can quickly get visually overwhelming. It is therefore often useful to compute covariances between the variables instead. The covariance matrix can then easily be visualized as a heatmap. A heatmap is effectively a pseudocolor plot with labelled rows and columns (i.e., a pseudocolor plot based on a pandas DataFrame rather than a matrix). The DataFrame does not have to be square or symmetric (but, in the context of a covariance matrix, it is both).
# 
# In this exercise, you will view the covariance matrix between the continuous variables in the auto-mpg dataset. You do not have to know here how the covariance matrix is computed; the important point is that its diagonal entries are all 1s, and the off-diagonal entries are between -1 and +1 (quantifying the degree to which variable pairs vary jointly). It is also, then, a symmetric matrix.
# 
# Instructions
# 
# 1-Print the covariance matrix cov_matrix to examine its contents and labels. This has been done for you.
# 2-Plot the covariance matrix cov_matrix using sns.heatmap()

# In[40]:


auto_heat =auto[['mpg', 'hp', 'weight', 'accel', 'displ']]
cov_matrix = auto_heat.cov()
print(cov_matrix)
sns.heatmap(cov_matrix)
plt.show()


#  # Visualizing time series
#  
#  Let load the stocks.cvs file like 'data/stocks.csv', and use Date column as index column
#  
# 1-Plot the aapl time series in blue with a label of 'AAPL'.
# 2-Plot the ibm time series in green with a label of 'IBM'.
# 3-Plot the csco time series in red with a label of 'CSCO'.
# 4-Plot the msft time series in magenta with a label of 'MSFT'.
# 5-Specify a rotation of 60 for the xticks with plt.xticks().
# 6-Add a legend in the 'upper left' corner of the plot.
# 

# In[67]:


stocks = pd.read_csv('data/stocks.csv', index_col='Date', parse_dates=True)
print(stocks.head())
print(stocks.loc['2013-12-31'])


# In[70]:


import matplotlib.pyplot as plt
aapl = stocks[['AAPL']]
ibm = stocks[['IBM']]
csco = stocks[['CSCO']]
msft = stocks[['MSFT']]

plt.plot(aapl, color='b', label='AAPL')
plt.plot(ibm, color='g', label='IBM')
plt.plot(csco, color='r', label='CSCO')
plt.plot(msft, color='m', label='MSFT')

plt.legend(loc='upper left')
plt.xticks(rotation=60)
plt.show()


# You can easily slice subsets corresponding to different time intervals from a time series. In particular, you can use strings like '2001:2005', '2011-03:2011-12', or '2010-04-19:2010-04-30' to extract data from time intervals of length 5 years, 10 months, or 12 days respectively.
# 
# Unlike slicing from standard Python lists, tuples, and strings, when slicing time series by labels (and other pandas Series & DataFrames by labels), the slice includes the right-most portion of the slice. That is, extracting my_time_series['1990':'1995'] extracts data from my_time_series corresponding to 1990, 1991, 1992, 1993, 1994, and 1995 inclusive.
# You can use partial strings or datetime objects for indexing and slicing from time series.
# For this exercise, you will use time series slicing to plot the time series aapl over its full 11-year range and also over a shorter 2-year range. You'll arrange these plots in a 2×1 grid of subplots
# 
# Instructions
# 
# 1-Plot the series aapl in 'blue' in the top subplot of a vertically-stacked pair of subplots, with the xticks rotated to 45 degrees.
# 2-Extract a slice named view from the series aapl containing data from the years 2007 to 2008 (inclusive). 
# 3-Plot the slice view in black in the bottom subplot.

# In[72]:


plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL 2001 to 2011')
plt.plot(aapl, color='blue')

view = aapl['2007':'2008']
plt.subplot(2,1,2)
plt.xticks(rotation=45)
plt.title('AAPL 2007 to 2008')
plt.plot(view, color='black')
plt.tight_layout()
plt.show()


# 1-Extract a slice named view from the series aapl containing data from November 2007 to April 2008 (inclusive). 
# 2-Plot the slice view in 'red' in the top subplot of a vertically-stacked pair of subplots with the xticks rotated to 45 degrees.
# 3-Reassign the slice view to contain data from the series aapl for January 2008.
# 4-Plot the slice view in 'green' in the bottom subplot with the xticks rotated to 45 degrees.

# In[73]:


view = aapl['2007-11':'2008-04']

plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL November 2007 to April 2008')
plt.plot(view, color='red')

view =aapl['2008-01']

plt.subplot(2,1,2)
plt.xticks(rotation=45)
plt.title('AAPL January 2008')
plt.plot(view, color='green')

plt.tight_layout()
plt.show()


# 1-Extract a slice of series aapl from November 2007 to April 2008 inclusive.
# 2-Plot the entire series aapl.
# 3-Create a set of axes with lower left corner (0.25, 0.5), width 0.35, and height 0.35. Pass these coordinates to plt.axes() as a list (all in units relative to the figure dimensions).
# 4-Plot the sliced view in the current axes in 'red'

# In[75]:


view =aapl['2007-11':'2008-04']

plt.plot(aapl)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')

plt.axes([0.25,0.5,0.35,0.35])

plt.plot(view, color='red')
plt.xticks(rotation=45)
plt.title('AAPL: 2007-11/2008-04')
plt.show()


# # Extracting a histogram from a grayscale image
# 1-Load data from the file '800px-Unequalized_Hawkes_Bay_NZ.jpg' into an array.'img/800px-Unequalized_Hawkes_Bay_NZ.jpg'
# 2-Display image with a color map of 'gray' in the top subplot.
# 3-Flatten image into a 1-D array using the .flatten() method.
# 4-Display a histogram of pixels in the bottom subplot.
# 5-Use histogram options bins=64, range=(0,256), and normed=True to control numerical binning and the vertical scale.
# 6-Use plotting options color='red' and alpha=0.4 to tailor the color and transparency.

# In[83]:


image = plt.imread('img/800px-Unequalized_Hawkes_Bay_NZ.jpg')

plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image, cmap='gray')

pixels= image.flatten()

plt.subplot(2,1,2)
plt.title('Normalized histogram')
plt.xlim((0,255))
plt.axis('off')
plt.hist(pixels, bins=64, range=(0,256), normed=True, color='red', alpha=0.4)

plt.show()


# # Cumulative Distribution Function from an image histogram
# A histogram of a continuous random variable is sometimes called a Probability Distribution Function (or PDF). The area under a PDF (a definite integral) is called a Cumulative Distribution Function (or CDF). The CDF quantifies the probability of observing certain pixel intensities.
# 
# Your task here is to plot the PDF and CDF of pixel intensities from a grayscale image. This time, the 2D array image will be pre-loaded and pre-flattened into the 1D array pixels .
#  -- The histogram option cumulative=True permits viewing the CDF instead of the PDF.
#  -- Notice that plt.grid('off') switches off distracting grid lines.
#  -- The command plt.twinx() allows two plots to be overlayed sharing the x-axis but with different scales on the y-axis.
#  
# 1-First, use plt.hist() to plot the histogram of the 1-D array pixels in the bottom subplot.
# 2-Use the histogram options bins=64, range=(0,256), and normed=False.
# 3-Use the plotting options alpha=0.4 and color='red' to make the overlayed plots easier to see.
# 4-Second, use plt.twinx() to overlay plots with different vertical scales on a common horizontal axis.
# 5-Third, call plt.hist() again to overlay the CDF in the bottom subplot.
# 6-Use the histogram options bins=64, range=(0,256), and normed=True.
# 7-This time, also use cumulative=True to compute and display the CDF.
# 7-Also, use alpha=0.4 and color='blue' to make the overlayed plots easier to see.
#  

# In[87]:


image = plt.imread('img/800px-Unequalized_Hawkes_Bay_NZ.jpg')

plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image, cmap='gray')

pixels= image.flatten()

plt.subplot(2,1,2)
pdf = plt.hist(pixels, bins=64, range=(0,256), normed=False, color='red', alpha=0.4)
plt.grid('off')
# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()

cdf = plt.hist(pixels, bins=64, range=(0,256), normed=True, cumulative=True, color='blue', alpha=0.4)
plt.grid('off')
plt.title('PDF & CDF of original imaga')
plt.show()


# # Equalizing an image histogram
# Histogram equalization is an image processing procedure that reassigns image pixel intensities. The basic idea is to use interpolation to map the original CDF of pixel intensities to a CDF that is almost a straight line. In essence, the pixel intensities are spread out and this has the practical effect of making a sharper, contrast-enhanced image. This is particularly useful in astronomy and medical imaging to help us see more features.
# 
# For this exercise, you will again work with the grayscale image of Hawkes Bay, New Zealand (originally by Phillip Capper, modified by User:Konstable, via Wikimedia Commons, CC BY 2.0). Notice the sample code produces the same plot as the previous exercise. Your task is to modify the code from the previous exercise to plot the new equalized image as well as its PDF and CDF.
# 
# The arrays image and pixels are extracted for you in advance.
# The CDF of the original image is computed using plt.hist().
# Notice an array new_pixels is created for you that interpolates new pixel values using the original image CDF.
# Instructions
# 
# 1-Define folllowing item 
# cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
# new_pixels = np.interp(pixels, bins[:-1], cdf*255)
# 2-Use the NumPy array method .reshape() to create a 2-D array new_image from the 1-D array new_pixels. The resulting new_image should have the same shape as image.shape.
# 3-Display new_image with a 'gray' color map to display the sharper, equalized image.
# 
# 4 Plot the PDF of new_pixels in 'red'. with using new_pixel
# 5-Use plt.twinx() to overlay plots with different vertical scales on a common horizontal axis.
# 6-Plot the CDF of new_pixels in 'blue'.

# In[94]:


cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)

new_image = new_pixels.reshape(image.shape)

plt.subplot(2,1,1)
plt.axis('off')
plt.title('Equalized image')
plt.imshow(new_image, cmap='gray')

plt.subplot(2,1,2)
pdf = plt.hist(new_pixels, bins=64, range=(0,256), normed=False, color='red', alpha=0.4)
plt.grid('off')
plt.title('PDF & CDF (equalized image)')

plt.twinx()
plt.xlim(0,256)

cdf = plt.hist(new_pixels, bins=64, range=(0,256), normed=True, cumulative=True, color='blue',alpha=0.4)
plt.axis('off')

plt.show()


# 1- load the image 'img/hs-2004-32-b-small_web.jpg'
# 2-Display image in the top subplot of a 2×1 subplot grid. Don't use a colormap here.
# 3-Extract 2-D arrays of the RGB channels: red, blue, green
#     red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
# 4-Flatten the 2-D arrays red, green, and blue into 1-D arrays
# 5-Display three histograms in the bottom subplot: one for red_pixels, one for green_pixels, and one for blue_pixels. For each, use 64 bins and specify a translucency of alpha=0.2.

# In[100]:


image = plt.imread('img/hs-2004-32-b-small_web.jpg')

plt.subplot(2,1,1)
plt.title('original image')
plt.axis('off')
plt.imshow(image)

#Extract 2-D arrays of the RGB channels: red, blue, green
red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

red_pixels = red.flatten()
blue_pixels = blue.flatten()
green_pixels= green.flatten()

plt.subplot(2,1,2)
plt.title('Histogram from color image')
plt.xlim(0,256)
plt.hist(red_pixels, color='red', bins=64, normed=True, alpha=0.2)
plt.hist(blue_pixels, color='blue', bins=64, normed=True, alpha=0.2)
plt.hist(green_pixels, color='green', bins=64, normed=True, alpha=0.2)

plt.show()


# 
