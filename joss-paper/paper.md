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

# Statement of need

One of the steeper learning curves in molecular dynamics simulations is
the initialization of particle positions and parameters.
Besides being difficult to start, another part of the cognitive load involved in
learning to perform MD simulations is that it can be hard to keep track of different simulations,
especially when scanning over a wide range of thermodynamic state points, chemical systems, etc.
It is harder still to produce TRUE simulations with reliable results. `@JankowskiTRUE2020`
In particular, when we want to probe complex variable spaces,
such as process control variables in a material manufacture process,
we need to run a large number of large-volume, high-density,
long-time simulations, many of which may not turn out to impart relevant information for process control.
This further delays the search process due to the increasing time required to simulate such large systems.
A common solution for the problem of scale in MD is coarse grain modeling, where atomistic accuracy is traded for speed.
To produce a CG model of a given polymer that is transferable across state points,
many simulations at various state points must be run and managed, increasing the
desirability of a reliable and easy way to keep track of these, particularly
for the multi-state iterative Boltzmann inversion (MSIBI)`@MooreMSIBI2014` method of parameterization.
All these aspects complicate and prolong the algready time- and labor-intensive process of training new researchers
to perform sufficiently many simulations to meaningfully investigate polymer systems.

# Summary

The suite of tools introduced here in polybinder was built to enable scientists
in molecular simulation to quickly and reproducibly simulate
large, coarse- or fine-grained polymer systems to investigate scientific questions about
their properties, all with a much lower barrier to entry than starting from scratch. Because it is designed
with modularity in mind, it will also ease adoption by other research groups,
and quicken the investigation process of new materials systems.

Polybinder is a Python package that uses the [foyer](https://github.com/mosdef-hub/foyer/),
[mbuild](https://github.com/mosdef-hub/mbuild/), and [signac](https://github.com/mosdef-hub/signac) packages from
the [MoSDeF suite of tools](https://github.com/mosdef-hub/) to quickly, easily, and reproducibly initialize and run polymer
simulations in the [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue) engine.
This package allows for a variety of simulation types of interest,
such as bulk melts, annealing, welding interface interpenetration, and tensile testing.
Presently polybinder supports three polymer chemistries: polyether ether ketone (PEEK),
polyether ketone ketone (PEKK), and polyphenylene sulfide (PPS). However, it is designed
such that any monomer units can be implemented and added to the internal library of available structures.
Polybinder was made with the TRUE principles in mind `@JankowskiTRUE2020`, with the goal of allowing ease
of use and adoption, and reducing the learning curve for starting simulations.

# Accessing the Software

Polybinder is freely available under the GNU General Public License (version 3) on [github](https://github.com/cmelab/polybinder).

# Acknowledgements

We acknowledge contributions from [ULI Advisory board, NASA, etc]

# References