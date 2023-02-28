---
title: 'Polybinder: A Python package for streamlined polymer molecular dynamics'
tags:
  - Python
  - materials science
  - molecular dynamics
  - polymers
  - HOOMD-blue
authors:
  - name: Chris Jones
    orcid: 0000-0002-6196-5274
    equal-contrib: true
    affiliation: 1
  - name: Rainier Barrett
    orcid: 0000-0002-5728-9074
    equal-contrib: true
    affiliation: 1
  - name: Eric Jankowski
    orcid: 0000-0002-3267-1410
    corresponding: true 
    affiliation: 1
affiliations:
 - name: Boise State University, Boise, ID, USA
   index: 1
date: 01 January 2001
bibliography: paper.bib

---

# Summary

Polybinder is a Python package that uses the Foyer, mbuild, and signac packages from
the MoSDeF suite of tools to quickly, easily, and reproducibly initialize and run polymer
simulations in HOOMD-blue. This package allows for a variety of simulation types of interest,
such as slab melts, annealing, welding interfaces, and tensile testing.
Presently Polybinder supports three polymer chemistries: polyether ether ketone (PEEK),
polyether ketone ketone (PEKK), and polyphenylene sulfide (PPS), but it is designed
such that any monomer units can be implemented and added to the internal library of available
structures.
Polybinder was made with the TRUE principles in mind [cite TRUE paper], with the goal of allowing ease
of use and adoption, and reducing the learning curve for starting simulations.

# Statement of need

One of the steeper learning curves in molecular dynamics simulations is
the initialization of particle positions and parameters.
It is hard to start and hard to keep track of different simulations,
especially when scanning over various state points.
It's harder still to produce TRUE simulations with reliable reults. [cite some reproducibility papers]
In particular, when we want to probe process control variables in a material
manufacture process, we need to look at a large number of large-volume, high-number,
long-time simulations, many of which will not be relevant, complicating the search.
Because of this, the use of coarse grain modeling[cite CG?] for polymer simulations
[cite some of those] is becoming more common.
To produce a CG model of a given polymer that is transferable across state points,
many simulations at various state points must be run and managed, increasing the
desirability of a reliable and easy way to keep track of these, particularly
for the multi-state iterative Boltzmann inversion (MSIBI) method of parameterization.

The suite of tools introduced here in polybinder enables newer scientists
in molecular simulation to quickly and reproducibly simulate
large, coarse-grained, complex polymer systems to investigate scientific questions about
their properties with a much lower barrier to entry than before. Because it is designed
with modularity in mind, it will also ease adoption by other research groups,
quickening the investigation process.


Why did we make polybinder?
* CG of polymers is hard
* Even just setting up sims to train a CG FF is hard
* Polybinder helps with that

# Accessing the Software

Polybinder is freely available under the GNU General Public License (version 3) on [github](https://github.com/cmelab/polybinder).

# Mathematics

Maybe skip this section? Or explain IBI?

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from [ULI Advisory board, NASA, etc]

# References

References go here.