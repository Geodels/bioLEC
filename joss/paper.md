---
title: 'bioLEC: A Python package to measure Landscape Elevational Connectivity'
tags:
  - Python
  - landscape evolution
  - geomorphology
  - hydrology
  - surface processes
  - stratigraphy
authors:
 - name: Tristan Salles
   orcid: 0000-0001-6095-7689
   affiliation: "1"
 - name: Patrice Rey
   orcid: 0000-0002-1767-8593
   affiliation: "1"
affiliations:
 - name: School of Geosciences, The University of Sydney, Australia
   index: 1
date: 2 June 2019
bibliography: paper.bib
---

# Summary

Understanding how biodiversity formed and evolved is a key challenge in evolutionary and ecological biology [@Newbold:2016].

Despite a theoretical consensus on how best to measure biodiversity from a biological perspective (_i.e._ number of species, length of all branches on the tree of life of a species, and differences in allele and genotype frequencies within species) standardised and cost-effective methods for assessing it on a broad range of scales are lacking [@Chiarucci:2011].

In mountainous landscapes long known for playing a significant role in evolution [@Wallace:1860, @Elsen:2015], one approach consists in quantifying and measuring abiotic landscape properties. Indeed, complex mountainous landscapes provide species with a rich variety of environmental conditions over restricted surface areas (_e.g._ range of temperature, solar irradiation, wind exposure, moisture and rainfall, soils thickness and composition, etc) [@Hoorn:2013].

Estimates of some of theses landscape abiotic properties are already available through standard software such as _ArcGIS_ or _QGIS_ [@Etherington:2016] and in more specific mountainous landscape focussed packages such as _LSD Topo Tools_ [@Clubb:2017] or _pyBadlands_ [@Salles:2018]. As an example, these tools include functions to assess slope and slope curvature from which soil thickness can be predicted, slope azimuth (which controls solar radiation); the number of catchments and their size (landscape fragmentation), the hydrological connectivity and drainage density (transfer of water and nutrients); the hypsometry (_i.e._ elevation vs surface area), or the direction and rate of divide migration.

# `bioLEC`

![An example of LEC map obtained for a specific elevation surface (left) illustrating the region of high and low connectivity as well as the distribution of resulting LEC values versus elevation range.\label{fig:example}](Fig1.jpg)

In 2016, a new metric called the _Landscape Elevational Connectivity_ (**LEC**) was proposed to estimate biodiversity in mountainous landscape [@Bertuzzo:2016]. It efficiently measures the landscape resistance to migration and is able to account for up to 70% of biodiversity predicted by meta-community models [@Bertuzzo:2016].

**bioLEC** is a Python package designed to quickly calculate for any mountainous landscape surface and species niche width its associated **LEC** index. From an elevational fitness perspective, all migratory paths on a flat landscape are equal. This is not the case on a complex landscape where migration can only occur along a network of corridors providing species with their elevational requirements. Hence, predicting how species will disperse across a landscape requires a model of migration that takes into account the physical properties of the landscape, the species fitness range, as well as evolving environmental conditions.

**LEC** measures the landscape resistance to migration by calculating the connectivity between points in the landscape based on _scikit-image_ Dijkstra’s algorithm [@vanderWalt:2014], the higher the connectivity the lesser the landscape resistance to migration. It quantifies the closeness of any of these points to all others at similar elevation by integrating specific fitness range and represents a robust proxy for the biodiversity a landscape can support [@Bertuzzo:2016]. It measures how easily a species living in a given patch can spread and colonise other patches. As explained above it is assumed to be elevation-dependent and it depends on how often a species adapted to a given elevation needs to travel outside its optimal elevation range when moving from its patch to any other in the landscape.

**bioLEC** package can be used in serial or parallel to evaluate biodiversity patterns of a given landscape. It is an efficient and easy-to-use tool to quickly assess the capacity of landscape to support biodiversity and to predict species dispersal across mountainous landscapes.


# Acknowledgements

The authors acknowledge the Sydney Informatics Hub and the University of Sydney's high performance computing cluster Artemis for providing the high performance computing resources that have contributed to the research described within this paper.

# References
