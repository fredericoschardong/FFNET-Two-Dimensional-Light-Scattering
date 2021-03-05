# FFNET-Two-Dimensional-Light-Scattering
Python implementation of a feed-forward neural network to predict the size of small particles based on their two-dimensional light scattering patterns.

"Clouds and their constituents (mostly small particles) have relationships with climate change. Better understanding and characterization of cloud particles is essential for the creation of better climate models for better understanding of climate and its dynamics/changes. Widely used cloud probes has resolution limitations, and two-dimensional light scattering patterns (2DLS patterns) of particles is an increasingly more promising approach for characterising (i.e. obtaining sizes, aspect ratios, shapes, concavities, and roughness of) cloud particles." [[1]](#1).

A brief analysis of the results is [provided in Portuguese](https://github.com/fredericoschardong/FFNET-Two-Dimensional-Light-Scattering/blob/master/report.pdf). It was submitted as an assignment of a graduate course named [Connectionist Artificial Intelligence](https://moodle.ufsc.br/mod/assign/view.php?id=2122514) at UFSC, Brazil.

In short, the light intensity of laser measurements is fed to the FF network which tries to learn and predict the size of ice particles. This code uses Zernike polynomials to be rotation invariant and achieves `R²=0.966`.

![](https://raw.githubusercontent.com/fredericoschardong/FFNET-Two-Dimensional-Light-Scattering/master/result.png "")

## References
<a id="1">[1]</a> 
Salawu, E. O. (2015).
Development of Computational Models for Characterizing Small Particles Based on theirTwo-Dimensional Light Scattering Patterns.
MA thesis.
University of Hertfordshire, UK.
