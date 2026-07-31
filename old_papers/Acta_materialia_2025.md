%% 
%% Copyright 2007-2024 Elsevier Ltd
%% 
%% This file is part of the 'Elsarticle Bundle'.
%% ---------------------------------------------
%% 
%% It may be distributed under the conditions of the LaTeX Project Public
%% License, either version 1.3 of this license or (at your option) any
%% later version.  The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% The list of all files belonging to the 'Elsarticle Bundle' is
%% given in the file `manifest.txt'.
%% 
%% Template article for Elsevier's document class `elsarticle'
%% with harvard style bibliographic references

\PassOptionsToPackage{square,comma,numbers,sort&compress}{natbib}

\documentclass[preprint,12pt]{elsarticle}

%% Use the option review to obtain double line spacing
%% \documentclass[authoryear,preprint,review,12pt]{elsarticle}

%% Use the options 1p,twocolumn; 3p; 3p,twocolumn; 5p; or 5p,twocolumn
%% for a journal layout:
%% \documentclass[final,1p,times,authoryear]{elsarticle}
%% \documentclass[final,1p,times,twocolumn,authoryear]{elsarticle}
%% \documentclass[final,3p,times,authoryear]{elsarticle}
%% \documentclass[final,3p,times,twocolumn,authoryear]{elsarticle}
%% \documentclass[final,5p,times,authoryear]{elsarticle}
%% \documentclass[final,5p,times,twocolumn,authoryear]{elsarticle}

%% For including figures, graphicx.sty has been loaded in
%% elsarticle.cls. If you prefer to use the old commands
%% please give \usepackage{epsfig}

%% The amssymb package provides various useful mathematical symbols
\usepackage{amssymb}
%% The amsmath package provides various useful equation environments.
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{colortbl}
\usepackage{amssymb}
\usepackage{array}
\usepackage{pifont} % For check and cross marks
\usepackage{subcaption}
\usepackage{upgreek}
\usepackage{float}
\usepackage{placeins} 
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,        % Enable colored links
    linkcolor=blue,         % Color for internal links (e.g., sections, figures)
    citecolor=red,          % Color for citations
    urlcolor=magenta,       % Color for URLs
    filecolor=cyan          % Color for file links
}

\usepackage[table]{xcolor} % Include xcolor for coloring
% Define a custom dark green color
\definecolor{darkgreen}{RGB}{0,128,0} % Adjust RGB values for your desired shade
% Command for bold dark green tick
\newcommand{\bigtick}{\textbf{\textcolor{darkgreen}{\checkmark}}}

\usepackage{booktabs}
\usepackage{rotating}
\usepackage{tabularray}
\usepackage{multirow}
\usepackage{multicol}

\captionsetup[table]{labelsep=space, labelfont=bf, textfont=normalfont, justification=justified}

\renewcommand{\figurename}{Fig.} 
\newcommand{\figref}[1]{\hyperref[#1]{Fig.~\ref*{#1}}}
\newcommand{\tabref}[1]{\hyperref[#1]{Table~\ref*{#1}}}




%% The amsthm package provides extended theorem environments
%% \usepackage{amsthm}

%% The lineno packages adds line numbers. Start line numbering with
%% \begin{linenumbers}, end it with \end{linenumbers}. Or switch it on
%% for the whole article with \linenumbers.
%% \usepackage{lineno}

\journal{Acta Materialia}

\begin{document}

\begin{frontmatter}

%% Title, authors and addresses

%% use the tnoteref command within \title for footnotes;
%% use the tnotetext command for theassociated footnote;
%% use the fnref command within \author or \affiliation for footnotes;
%% use the fntext command for theassociated footnote;
%% use the corref command within \author for corresponding author footnotes;
%% use the cortext command for theassociated footnote;
%% use the ead command for the email address,
%% and the form \ead[url] for the home page:
%% \title{Title\tnoteref{label1}}
%% \tnotetext[label1]{}
%% \author{Name\corref{cor1}\fnref{label2}}
%% \ead{email address}
%% \ead[url]{home page}
%% \fntext[label2]{}
%% \cortext[cor1]{}
%% \affiliation{organization={},
%%            addressline={}, 
%%            city={},
%%            postcode={}, 
%%            state={},
%%            country={}}
%% \fntext[label3]{}

\title{Machine learning pipeline for Structure-Property modeling in Mg-alloys using microstructure and texture descriptors.} %% Article title

%% use optional labels to link authors explicitly to addresses:
%% \author[label1,label2]{}
%% \affiliation[label1]{organization={},
%%             addressline={},
%%             city={},
%%             postcode={},
%%             state={},
%%             country={}}
%%
%% \affiliation[label2]{organization={},
%%             addressline={},
%%             city={},
%%             postcode={},
%%             state={},
%%             country={}}

\author[label1]{Mahish K. Guru}
\author[label1]{Jan Bohlen}
\author[label1,label3]{Roland C. Aydin}
\author[label1,label2]{Noomane Ben Khalifa}

\affiliation[label1]{organization={Institute of Material and Process Design, Helmholtz-Zentrum Hereon},%Department and Organization
            % addressline={}, 
            city={Geesthacht},
            % postcode={21502}, 
            % state={},
            country={Germany}}
\affiliation[label2]{organization={Institute for Production Technology and Systems, Leuphana Universität Lüneburg},%Department and Organization
            % addressline={}, 
            city={Lüneburg},
            % postcode={}, 
           % state={State One},
            country={Germany}}            
\affiliation[label3]{organization={Institute of Continuum and Materials Mechanics, Hamburg University of Technology (TUHH)},%Department and Organization
            % addressline={}, 
            city={Hamburg},
            % postcode={}, 
            % state={},
            country={Germany}}

%% Abstract
\begin{abstract}
%% Text of abstract
Identifying the relationships between material structure and mechanical properties has been crucial for accelerating the exploration of the material design space for advanced alloys. However, traditional approaches for magnesium (Mg) alloys often fall short in providing quantitative and broadly applicable structure-property linkages. To address this challenge, a comprehensive machine learning pipeline is presented for structure-property modeling in extruded Mg-alloys, leveraging both microstructure and texture descriptors derived from experimental data. The pipeline encompasses a robust workflow for data extraction from optical microscopy and X-ray diffraction, advanced image processing and deep learning techniques for microstructure binarization and grain statistics, and the computation of statistical descriptors including n-point spatial correlations, gram matrices for microstructure, and generalized spherical harmonics (GSH) for texture. Dimensionality reduction techniques such as principal component analysis (PCA), isomap, and autoencoders are employed to manage the high-dimensionality of the descriptor space. Subsequently, non-linear regression models – Gaussian Process, XGBoost, and Multi-Layer Perceptron regressors – are evaluated to predict mechanical properties, specifically strain hardening exponent ($\mathbf{n}$) and yield stress ($\mathbf{\sigma_{y}}$). Our results demonstrate that XGBoost consistently outperforms other regressors, achieving a notably low mean absolute percentage error (MAPE) of 6.67\% for strain hardening exponent and 7.01\% for yield stress, using a combination of PCA-reduced 3-point spatial correlations and isomap-reduced gram matrices as microstructure descriptors, and isomap-reduced GSH coefficients as texture descriptors at a $150 \,\upmu \text{m}$ length scale. Shapley Additive exPlanations (SHAP) analysis further reveals that texture descriptors and aspect ratio distribution are the most influential features in predicting mechanical properties. This established ML framework for structure-property modeling in Mg-alloys, surpasses state-of-the-art benchmarks and provides a valuable template for materials design and discovery.

\end{abstract}

%%Graphical abstract
\begin{graphicalabstract}
\includegraphics[width=\textwidth]{MahishVisualAbstract.png}
\end{graphicalabstract}

%%Research highlights
\begin{highlights}
\item Machine learning pipeline that utilizes statistical descriptors for representing microstructure and texture for extruded Mg-alloys.
\item Data extraction and preprocessing to enrich and engineer features, forming a coherent dataset.
\item Dimensionality reduction (PCA, Isomap, Autoencoder) applied to the descriptors for lower order Structure-Property linkages.
\item GP, XGBoost, and MLP as non-linear regressors for property prediction in Mg-alloys.
\item Outperforms state-of-the art benchmarks for property prediction based on training and testing on experimental data.
\end{highlights}

%% Keywords
\begin{keyword}
%% keywords here, in the form: keyword \sep keyword

%% PACS codes here, in the form: \PACS code \sep code

%% MSC codes here, in the form: \MSC code \sep code
%% or \MSC[2008] code \sep code (2000 is the default)

Structure-Property \sep Magnesium alloys \sep Microstructure \sep Texture \sep Descriptor \sep Grain boundary \sep Orientation distribution function \sep Property prediction \sep Machine learning \sep Deep learning

\end{keyword}

\end{frontmatter}

%% Add \usepackage{lineno} before \begin{document} and uncomment 
%% following line to enable line numbers
%% \linenumbers

%% main text
%%

%% Use \section commands to start a section
\section{Introduction}
\label{sec1}

The integration of machine learning (ML) into material science has revolutionized the field, particularly in the domains of material and process design \citep{RICKMAN2019473}. Machine learning algorithms, with their ability to analyze vast datasets and identify complex process-structure-property relationships, are pshowing critical importance in synthesizing new materials \citep{MOBARAK2023100523}. Simulation-driven material design has long played a crucial role in establishing structure-property relationships. Now, tremendous advancements in Integrated Computational Materials Engineering (ICME) are leveraging experimental data alongside material science principles and simulation to accelerate material design exploration \citep{YIWANG201942}. The advent of data-driven and hybrid techniques has enabled researchers and engineers to tailor material properties to meet specific performance requirements \citep{XZheng}. Due to the applications of polycrystalline metallic materials in aerospace, automotive, bio-materials, renewable energy, and electronics, machine learning has been widely used to uncover structure-property relationships in these materials. Machine learning-embedded workflows have been used to develop structure-property relationships in steels \citep{REN2023118954}, aluminum alloys \citep{LIU2024104187}, titanium alloys \citep{ZHAO2023145202}, copper alloys \citep{GUO2024147344}, and more.

Magnesium (Mg) alloys are another highly popular polycrystalline metallic material, used in aerospace, transportation, and biomedical applications due to their properties such as low corrosion resistance, high strength, and low density. The mechanical properties of wrought magnesium alloy ( produced through mechanical processes such as rolling, forging, and extrusion) can be tailored by setting the processing parameters appropriately, such as temperatures and processing times \citep{JAYASATHYAKAWIN2020909}. For these reasons, Mg-alloys have been extensively studied as functional materials in recent years as lightweight material for replacement of steel and aluminum in the aerospace and automobile industry. However, the widespread industrial adoption of Mg-alloys has been hindered by their low formability, as well as ability to adjust strength, resulting in the low yield strength (less than 300 MPa) of most commercial Mg-alloys \citep{WANG202427}. Consequently, significant efforts have been undertaken to enhance the strength of magnesium alloys by thoroughly understanding the effects of various fabrication processes. Extrusion has been employed as a forming technique for many polycrystalline metallic materials to produce fine-grained microstructures. For industrial Mg-alloys, it has been found that by adjusting the temperature and speed of extrusion, the tensile yield strength (TYS) can be significantly enhanced \citep{BEER2012304}.

Numerous experimental studies have been conducted to understand the effects of alloying and to correlate microstructure and textures with mechanical and formability properties of extruded Mg-alloys. The impact of rare earth elements, such as cerium (Ce) and neodymium (Nd), on the microstructure and texture evolution of magnesium-manganese (Mg-Mn) alloys during extrusion, and their subsequent influence on mechanical properties, has also been investigated. The study reveals that these elements weaken the typically strong recrystallization textures seen in the base alloy. Among them, neodymium proves to be the most effective in modifying the texture by reducing the formation of grains with orientations unfavorable for deformation. This modification leads to enhanced ductility and a reduction in asymmetric yield behavior, ultimately resulting in lower yield strength and ultimate tensile stress \citep{BOHLEN20107092}. In a study conducted by \citep{Nienaber2019}, direct extrusion of flat magnesium alloy bands was used to investigate the effects of alloying and processing on microstructure and texture development. The study focused on how extrusion speed influenced dynamic recrystallization, grain growth, and texture modification in alloys like AZ31, ZX10, and ZN10. By adjusting alloying elements such as Nd and Ca, the study achieved weaker basal textures, characterized by a less pronounced alignment of basal planes. This texture weakening significantly enhances mechanical properties like ductility and formability, particularly in biaxial tests, by allowing easier activation of slip systems. Another study investigated the effect of silver (Ag) alloying in magnesium (Mg) on microstructure and mechanical properties during indirect extrusion. It found that increasing Ag content enhanced tensile yield strength (TYS) and ultimate tensile strength (UTS) by promoting non-basal slip mechanisms, improving elongation. The development of a weak basal texture, where basal planes align with the extrusion direction, facilitated these mechanical improvements by enhancing ductility and reducing twinning during compression \citep{WIESE2021112}. 

While these experimental studies provide deep insights into the correlations between microstructure, textures, and mechanical properties of extruded Mg-alloys, their primary focus is to offer knowledge-based insights into material behavior by revealing dominant metallurgical mechanisms. However, these approaches do not provide a quantitative overview for a broader range of alloying or processing conditions. Additionally, these studies are often expensive and time-consuming, leading to prolonged material development timelines. Despite these challenges, experimental studies remain a crucial source of primary data. However, there is a pressing need for a more quantified approach to studying these correlations using machine learning based methodologies, enabling the development of more precise and efficient ways to analyze structure-property relationships \citep{XZheng}.

A comprehensive Gaussian process (GP)-based workflow was developed to predict the process-structure-property (PSP) linkages in metal additive manufacturing, specifically focusing on laser powder bed fusion processes using 316L stainless steel. The methodology involved creating functional GPs trained on a limited set of experimental and simulation data to predict melt pool characteristics, microstructure features, and mechanical properties. The study found that the GP-based framework could achieve approximately 95\% accuracy in predicting grain stress and strain for previously unseen process parameter combinations \citep{SAUNDERS2023103398}. Different machine learning methods have also been previously compared by \citep{FernandezZelaia2019ACS} in establishing structure-property linkages for high-contrast 3D elastic composites. They used finite element simulations to generate a dataset of voxelized microstructures and applied principal component analysis for dimensionality reduction. The study compared local/global and parametric/non-parametric approaches, with a focus on Gaussian Process (GP) models to predict the FE estimated effective stiffness. The findings revealed that the locally approximated Gaussian Process model achieved the highest accuracy, with a mean absolute error (MAE) of 1.87 GPa and a mean squared prediction error (MSPE) of 2.68 GPa. A reduced order structure-property (S-P) linkage framework for polycrystalline microstructures of $\alpha$-titanium was developed by \citep{PAULSON2017428} using 2-point spatial correlations, generalized spherical harmonics (GSH), and principal component analysis (PCA) for compact representation of microstructure. They tested this framework on data generated using crystal plasticity finite element (CPFE) simulation. The findings showed that the framework could predict elastic stiffness with an average error of about 0.2\% and yield strength with an average error of about 1.5\%. While all the aforementioned approaches successfully link microstructure features (and in one case, texture features \citep{PAULSON2017428}) to mechanical properties with high accuracy, these accuracies are primarily derived from testing on simulation datasets or a combination of simulation and experimental datasets. This raises the question of whether the simulation datasets are sufficiently grounded in real-world experimental data, which is crucial for accelerating material design exploration. This is particularly important for leveraging the experimental data for industrial Mg-alloys.

Machine learning has indeed also been applied to predict properties in Mg-alloys. An in-depth study was conducted by \citep{DongShuya} to predict the mechanical properties using experimental data collected from published literature. This data was used to establish a comprehensive database of descriptors, including processing parameters (such as rolling temperature, extrusion temperature, and extrusion ratio), alloying element properties, thermodynamic parameters. Finally, several ML models were employed to predict ultimate tensile strength (UTS), elongation (EL), yield strength (YS), and hardness (HV). Gradient Boosting Regression (GBR) achieved  high accuracy for UTS, YS, and HV achieving an $R^2$ value of more than 0.8 on the test set, although the prediction for EL was less satisfactory. Despite the adequate accuracies achieved, the descriptor space was primarily composed of thermal processing parameters and elemental properties, with no information about the microstructure or texture of the material. This now highlights the need for a structure-property (S-P) linkage approach that is independent of thermal processing parameters or alloying elemental properties. Such an approach would ensure a more robust and generalizable model, capable of accurately predicting mechanical properties across a wider range of conditions and compositions. Another ensemble-based approach utilizing shallow artificial neural networks (ANNs) has been applied by \citep{WIESE2023106566} for extruded magnesium-gadolinium (Mg-Gd) alloys to predict grain size, tensile yield stress, compressive yield stress, ultimate tensile strength, and hardness. The models were trained on a dataset where descriptors included various process parameters, such as extrusion velocity, temperature, and gadolinium content. However, the models exhibited limited generalizability, with their accuracy declining when predicting properties beyond the measured process parameter space. This limitation can be attributed to the insufficiency for processing parameters and alloying content as a complete indicator of these properties. Additionally, the models only utilized grain size as a descriptor of microstructure and the texture is only simply considered as 3 pole related intensities rather than the representative orientation distribution.


Therefore, in this paper, 
\begin{itemize}
    \item We develop a database and a generalized approach for extruded Mg-alloys that bases the prediction of mechanical properties entirely on descriptors of microstructure and texture, as illustrated in the pipeline in \figref{fig:paper_schematic}.
    \item  We investigate various statistical descriptors for the most accurate and efficient characterization of microstructure, the orientation distribution of the microstructure grains and grain statistics.
    \item We explore dimensionality reduction techniques to create a reduced-order structure-property (S-P) linkage.
    \item We examine various non-linear regression models to achieve the most accurate and efficient prediction of mechanical properties.
\end{itemize}

\section{Structure-Property linkage framework}
\label{sec2}


\begin{figure*}[ht]
    \centering
    \includegraphics[width=\textwidth]{Figures/Paper_Schematic.png}
    \caption{Pipeline for Structure-Property modeling of Mg-alloys with 5 key modules: 1. Data extraction, input-output database creation, 2. Pre-processing the data for augmentation, information enrichment and feature engineering, 3. Microstructure characterization using statistical descriptors, 4. Dimensionality reduction of the calculated descriptors and 5. Structure-Property modeling to predict mechanical properties from the input lower order descriptors.}
    \label{fig:paper_schematic}
\end{figure*}


To develop a quantified and reduced-order approach for the structure-property relationship, efficient characterization, dimensionality reduction, and model selection are essential components of the pipeline. This necessitates a thorough investigation of these components to tailor the approach to specific use cases while maintaining generalizability across different use cases or material types. For instance, our pipeline utilizes experimental data from various extruded Mg-alloys, following a specific protocol for data on microstructure and texture features. However, efforts are made to achieve a more generalized structure-property approach for characterization, dimensionality reduction, and model selection, applicable not only to one specific data protocol but also to other widely available data protocols. To this end, we made a conscious decision to accommodate optical microscopy (OM) images for micrographs (with minimal effort required to extend the approach to scanning electron microscopy (SEM) images) and X-ray diffraction (XRD) data for texture (which can also be extended to electron backscatter diffraction (EBSD) data). These data types are more accessible and economical to acquire, yet they contain sufficient information indicative of the mechanical properties. By leveraging OM and XRD data, we ensure that the pipeline remains cost-effective and widely applicable across diverse datasets. This adaptation enhances the robustness and applicability of the method across various material systems and alloying elements, making it independent of specific compositions \citep{Bock2019}.

Microstructure information is extracted as the first part of the data extraction block (\figref{fig:paper_schematic}) from the experimental data using specific routines to obtain the micrographs. These micrographs, which are images obtained through metallography of the sample and observed under OM, reveal grain boundaries and grain structure. They often include a scale (generally in $\upmu m$) for spatial and scale perspective \citep{DIGIANFRANCESCO2017197}. More information about this data is given in Section \ref{sec3}. In the next stage of the pipeline, these micrographs undergo preprocessing to maintain uniform scale by cropping and tiling, ensuring data uniformity and coherence in representation, which is essential for making automated predictions using machine learning. 

For a quantified structure-property (S-P) linkage, raw microstructure images cannot effectively characterize the microstructure due to the presence of a lot of unimportant information in direct pixel-to-pixel comparisons. To address this, the images are binarized to create a clearer and more computer-readable format. This process is particularly advantageous for machine learning applications as it reduces noise, and enriches the information about grain boundary structure and topology, making it easier for algorithms analyze the information \citep{GORYNSKI2023119106}. Workflow used for microstructure binarization for our this pipeline is discussed in Section \ref{sec3_binarization}. In this pipeline, we focus on single-phase Mg-alloys, treating any other phase as an inclusion with a pixel value of 1. Additionally, grains are segmented using the binarized images of the microstructure to obtain statistics for grain size and grain aspect ratios. These statistics are then represented in the form of binned histograms. Conventionally, only these statistics have been employed to characterize microstructure features \citep{EOHall_1951,Petch1953TheCS}. However, they cannot fully capture the key features of grain morphology, which are crucial for property prediction applications. To address this, extensive research has been conducted to develop algorithms that calculate statistical descriptors for microstructure characterization \citep{Berryman1985MeasurementOS,SINGH2008104,Torquato}. An n-point spatial correlation function (or n-point statistics) is a statistical descriptor used to capture the spatial arrangement of grains and grain boundaries in polycrystalline materials. It can represent complex patterns and arrangements, including long-range patterns and connectivity features \citep{Cecen}. However, this method requires higher-dimensional representation and is more susceptible to noise in the data. In contrast, a Convolutional Neural Network (CNN) based descriptor, gram matrices are generally used for a more compact representation of complex spatial relationships of grains and grain boundaries in a polycrystalline microstructure. They provide direct geometric insights into the microstructure features, offering a description of the overall geometry of the grain boundaries in the microstructure image. This method also helps capture any anisotropy present in the material \citep{Lubbers, Seibert2023ReconstructingMF}. Spatial correlations capture local spatial arrangements, while gram matrices provide a global view of grain boundary features. The implementation of both descriptors is discussed in Section \ref{sec4}. Considering the advantages and disadvantages of grain statistics, n-point spatial correlations, and gram matrices, this research will evaluate their individual and combined performance in predicting mechanical properties during the extrusion of Mg-alloys.

Another part of the first phase of the S-P pipeline shown in \figref{fig:paper_schematic} is the extraction of texture description from the experimental data. This texture description is available in the form of metadata of pole figures, which are obtained from X-ray diffraction (XRD) experiments using a Malvern-Panalytical file structure (.xrdml). X-ray diffraction is a crucial tool for studying the atomic structure of polycrystalline materials. XRD is based on the interference of X-ray waves elastically scattered by a series of atoms oriented along particular lattice plains in a crystal \citep{Wagner1999}. XRD is used for obtaining pole figures, which are essential for determining if a material exhibits a texture, i.e., the orientation of the grains in the material. Pole figures display the accumulation of poles around specific directions relevant to the material’s processing, such as the normal direction (ND), extrusion direction (ED), and transverse direction (TD). For instance, in rolled metals, certain crystallographic directions may align preferentially due to the deformation processes involved during manufacturing \citep{Xiong1984}. These crystallographic orientations of grains are very indicative of the mechanical performance of the material. While pole figures are useful for visualizing these orientations and making quick assessments, they lack the depth of information needed for a data-driven approach to property prediction. Therefore, pole figures are further processed to calculate the Orientation Distribution Function (ODF), which captures the number of grains oriented in specific directions, expressed in terms of Euler angles \citep{BUNGE19811}. The ODF is more suitable for characterizing complex orientations and enables more detailed orientation feature extraction using machine learning. When represented in Bunge-Euler space $\left\{ g = \left( \varphi_1, \Phi, \varphi_2 \right) \, \middle| \, 0 \leq \varphi_1 < 2\pi, \ 0 \leq \Phi \leq \pi/2, \ 0 \leq \varphi_2 < \pi/6 \right\}$ with 5-degree increments, the ODF is generally high-dimensional data. For compactness, ODFs are most commonly represented in terms of expansion coefficients of GSH. More specifics about the implementation for converting pole figures to ODFs and then to GSH coefficients in our pipeline will be explained in Section \ref{sec5}. Further details about the texture descriptors and the calculation of GSH coefficients can be found in Section \ref{sec5}. It is also important to note that setting the upper limit for truncating the series expansion of GSH, L = 22, is sufficient to smoothly capture the orientation density \citep{EGHTESAD2018418}. However, this still results in more than 2000 coefficients, which is a high number for our intended black-box modeling \citep{BUNGE19811}.

Descriptors used in the pipeline for microstructure and texture are high-dimensional data. With high-dimensional data comes the curse of dimensionality, which makes it difficult for machine learning algorithms to learn meaningful patterns in these descriptors, as they can overfit to unimportant patterns \citep{Keogh2010}. Therefore, dimensionality reduction techniques such as PCA and manifold learning are employed to identify principal dimensions for n-point statistics and gram matrices used as microstructure descriptors. Additionally, Autoencoders and manifold learning are applied to reduce the dimensions of the direct ODF descriptions and the compact representations with GSH coefficients. In section \ref{sec6}, the implementation for these dimensionality reduction techniques are discussed. Furthermore, to determine the most effective dimensionality reduction method for our processed data types, these techniques are compared against each other based on their performance in predicting mechanical properties.

In this study, data extracted from the experimental sets for mechanical properties is in the form of stress-strain curves, which is characterized into scalar values that best represent the mechanical performance of the alloy using specific routines. These scalar values include Yield Strength ($\mathbf{\sigma_{y}}$), Ultimate Strength ($\mathbf{\sigma_{max}}$), Strain Hardening Exponent ($\mathbf{n}$), Uniform Strain ($\mathbf{\varepsilon_{pu}}$), and Fracture Strain ($\mathbf{\varepsilon_{pf}}$). These values are crucial for the material design workflow in both practical and industrial stages of material development and are discussed in detail in Section \ref{sec3_property}. They are used as labels for the black box modeling of phase 5. Structure-property models (see \figref{fig:paper_schematic} of the pipeline. We investigate the performance of non-linear regression techniques, including Multi-Layer Perceptron (MLP), XGBoost (based on random forest regression), and Gaussian Process (GP) Regressor, to model the complex relationships between features (microstructure and texture descriptors) and the labels (mechanical properties). These models are evaluated for their ability to efficiently and accurately predict mechanical properties, facilitating efficient industrial material design process for extruded Mg-alloys.

\section{Dataset and Materials}
\label{sec3}

For every supervised machine learning-based pipeline, a coherent and well-labeled dataset with uniform structure is the backbone. When modeling complex non-linear relationships between data and labels, the accuracy of the model depends significantly more on the quality and coherence of the input data than on the preprocessing and selection techniques \citep{Wang2023}. In this section, we will detail the materials used in this study, specifically the Mg-alloys, along with their sources and the specific data collected for each alloy. Furthermore, we will elaborate on the preprocessing techniques performed on the data, such as microstructure binarization, grain state analysis, ODF characterization, and GSH coefficient calculation for texture data. The methods of implementation for these preprocessing steps will be thoroughly discussed. Additionally, we will address the data labels used in this pipeline, specifically the stress-strain data along with the scalar values and their significance in the context of material design and property prediction.


\subsection{Material data}

Data were extracted for various Mg-alloys from previous experimental studies aimed at understanding the effect of the extrusion process on microstructure, texture, and consequently, mechanical properties. The selected studies were published and met specific criteria to ensure uniformity in process variables, namely extrusion velocity ($\mathbf{v_{ext}}$) and extrusion temperature ($\mathbf{T_{ext}}$). Further criteria for the availability of microstructure information as images from OM (optionally, also from EBSD) and availability texture descriptions in the form of raw data from XRD (optionally, from EBSD) for pole figures. Finally, stress-strain data (.csv files) for each specimen, for which the aforementioned data were available, were included to create a coherent, well-structured, and labeled dataset, as shown in \tabref{tab:matdata}.


\begin{table}[ht]
\centering
\caption{\newline Material data with Mg-alloy names and corresponding extrusion process parameters, type of microstructure and texture data available, the range of mechanical properties \citep{nienaber_einfluss_der_2023, CANOCASTILLO2020139527, Nienaber2019, KurzG, Harmuth, cryst12081036}.}
\label{tab:matdata}
\renewcommand{\arraystretch}{1.4}
\resizebox{\linewidth}{!}{
\begin{tabular}{l >{\centering\arraybackslash}p{2cm} >{\centering\arraybackslash}p{2cm} >{\centering\arraybackslash}p{4cm} >{\centering\arraybackslash}p{4cm} >{\centering\arraybackslash}p{2.2cm} >{\centering\arraybackslash}p{2.2cm} >{\centering\arraybackslash}p{2.2cm} >{\centering\arraybackslash}p{2.2cm} >{\centering\arraybackslash}p{2.2cm}}
\hline
\upmulticolumn{Alloy Name } & \begin{tabular}{c} 
$\mathbf{v_{ext}}$ \\ 
$(\mathrm{mm/s})$
\end{tabular} & \begin{tabular}{c} 
$\mathbf{T_{ext}}$ \\ 
$(\mathrm{^\circ\mathrm{C}})$
\end{tabular} & \begin{tabular}{c} 
Microstructure \\ 
$(IMG = [a_{i,j}]_{i,j \in M})$
\end{tabular} & \begin{tabular}{c} 
Texture \\ 
$(ODF = f(\varphi_1, \Phi, \varphi_2))$
\end{tabular} & \begin{tabular}{c} 
$\mathbf{{\sigma}_{y}}$ \\ 
$(\mathrm{MPa})$
\end{tabular} & \begin{tabular}{c} 
$\mathbf{{\sigma}_{max}}$ \\ 
$(\mathrm{MPa})$
\end{tabular} & \begin{tabular}{c} 
${n}$ \\ 
\end{tabular} & \begin{tabular}{c} 
$\mathbf{{\varepsilon}_{pu}}$ \\ 
\end{tabular} & \begin{tabular}{c} 
$\mathbf{{\varepsilon}_{pf}}$ \\ 
\end{tabular} \\
\hline 
& & & & & & & & & \\
AZ31             & $0.6, 2.4$ & $200-500$ & OM, EBSD & XRD, EBSD & $143-184$ & $240-278$ & $0.13-0.23$ & $0.10-0.18$ & $0.13-0.25$ \\
AZ31(HT10min450) & $0.6, 2.4$ & $200-500$ & OM, EBSD & XRD, EBSD & $140-180$ & $245-265$ & $0.14-0.23$ & $0.13-0.17$ & $0.17-0.24$ \\
ME21             & $0.75-7.5$ & $300-450$ & OM       & XRD       & $154-237$ & $228-270$ & $0.09-0.19$ & $0.08-0.15$ & $0.14-0.22$ \\
Mg-2Gd           & $0.5-2$    & $350-450$ & OM       & XRD       & $87-208$  & $179-226$ & $0.10-0.31$ & $0.07-0.26$ & $0.09-0.47$ \\
Mg-2Gd-0.5Mn     & $0.5-2$    & $350-450$ & OM       & XRD       & $104-184$ & $180-216$ & $0.13-0.23$ & $0.07-0.19$ & $0.09-0.31$ \\
Mg-2Gd-1Mn       & $0.5-2$    & $350-450$ & OM       & XRD       & $105-183$ & $188-221$ & $0.13-0.23$ & $0.13-0.17$ & $0.17-0.30$ \\
Mg-5Gd           & $0.5-2$    & $350-450$ & OM       & XRD       & $91-193$  & $240-278$ & $0.12-0.28$ & $0.12-0.26$ & $0.14-0.35$ \\
Mg-5Gd-0.5Mn     & $0.5-2$    & $350-450$ & OM       & XRD       & $109-173$ & $198-225$ & $0.16-0.23$ & $0.13-0.19$ & $0.15-0.26$ \\
Mg-5Gd-1Mn       & $0.5-2$    & $350-450$ & OM       & XRD       & $114-187$ & $206-237$ & $0.15-0.22$ & $0.15-0.18$ & $0.21-0.27$ \\
Mg-10Gd          & $0.5-2$    & $350-450$ & OM       & XRD       & $132-232$ & $242-296$ & $0.16-0.22$ & $0.18-0.24$ & $0.22-0.32$ \\
Mg-10Gd-0.5Mn    & $0.5-2$    & $350-450$ & OM       & XRD       & $138-212$ & $248-286$ & $0.17-0.20$ & $0.18-0.20$ & $0.22-0.26$ \\
Mg-10Gd-1Mn      & $0.5-2$    & $350-450$ & OM       & XRD       & $154-232$ & $262-296$ & $0.16-0.19$ & $0.18-0.20$ & $0.24-0.26$ \\
Z1               & $2-7.5$    & $250-400$ & OM, EBSD & XRD, EBSD & $125-149$ & $216-235$ & $0.19-0.23$ & $0.07-0.09$ & $0.16-0.24$ \\
ZNd10            & $2-7.5$    & $250-400$ & OM, EBSD & XRD, EBSD & $79-282$  & $203-287$ & $0.08-0.33$ & $0.09-0.23$ & $0.22-0.48$ \\
ZNd10(HT10min450)& $2.4$      & $350$     & OM, EBSD & XRD, EBSD & $72$      & $197$     & $0.32$      & $0.22$      & $0.31$      \\
ZX10             & $2-7.5$    & $250-400$ & OM, EBSD & XRD, EBSD & $77-132$  & $194-225$ & $0.23-0.34$ & $0.13-0.25$ & $0.25-0.44$ \\
ZX10(HT10min450) & $0.6,2.4$  & $300$     & OM, EBSD & XRD, EBSD & $62-69$   & $184-189$ & $0.36-0.37$ & $0.25-0.27$ & $0.35-0.39$ \\
& & & & & & & & & \\
\hline
\end{tabular}
}
\end{table}

The data for a wrought magnesium alloy AZ31, which contains approximately 2.5\% to 3.5\% aluminum, 0.7\% to 1.3\% zinc, and a minimum of 0.2\% manganese, were extracted from a doctoral thesis. In this thesis, flat bands of AZ31 were processed at extrusion speeds of 0.6 and 2.4 mm/s and extrusion temperatures of 200, 250, 300, 350, 400, 450 and 500 $\mathrm{^\circ\mathrm{C}}$, with and without heat treatment (HT) post extrusion, and subsequently analyzed for their microstructure, texture, and mechanical properties. The mechanical properties were evaluated via uni-axial tensile testing \citep{nienaber_einfluss_der_2023}. Furthermore, similar data were collected for zinc-based magnesium alloys with additions of calcium (Ca) and neodymium (Nd) from a study conducted by \citep{CANOCASTILLO2020139527}. These alloys were extruded to obtain round bars under extrusion temperatures of 250, 300, 350 and 400 $\mathrm{^\circ\mathrm{C}}$ and extrusion velocities of 2, 5 and 7.5 mm/s. The three alloys are designated as follows: the Mg-Zn alloy is named Z1, the Mg-Zn-Ca alloy is named ZX10, and the Mg-Zn-Nd alloy is named ZNd10. Additional data for AZ31, Z1, ZX10, and ZNd10, as well as heat-treated ZX10 and ZNd10, were collected from a study that investigated the effect of varying processing parameters on the mechanical properties of flat bands of Mg-alloys at extrusion speeds of 0.6 and 2.4 mm/s \citep{Nienaber2019}. Data for ME21 alloys, a specialized aluminum-free magnesium alloy (Mg–2Mn–0.6Ce–0.3) was extracted from a study that investigated the effect of varying extrusion parameters on tailoring mechanical properties with respect to microstructure and texture. The extrusion temperature was varied at 300, 350, 400, and 450 $\mathrm{^\circ\mathrm{C}}$, and the extrusion speed was varied at 0.75, 1.4, 2.8, 5.5, and 7.5 mm/s \citep{KurzG}. The minimum and maximum tensile yield strengths for ME21 were found to be 228 MPa and 270 MPa, respectively (see \tabref{tab:matdata}). Another specialized class of Mg-alloys, which is very popular for lightweight and biomedical applications, are Mg-Gd alloys due to their corrosion resistance and high strength. Data for Mg-2Gd, Mg-5Gd, and Mg-10Gd were taken from a study by \citep{Harmuth}, which investigated the effect of extrusion rates (0.5, 1, and 2 mm/s) and extrusion temperatures (350, 400, and 450 $\mathrm{^\circ\mathrm{C}}$) on mechanical properties. This study achieved a range of tensile yield strengths ($\mathbf{{\sigma}_{y}}$) from 180 to 280 MPa and tensile maximum strengths ($\mathbf{{\sigma}_{max}}$) from 300 to 450 MPa. The final addition to the dataset includes data for Mg-Gd alloys with manganese (Mn). This data was taken from a study that investigated Mg-2Gd-0.5Mn, Mg-5Gd-0.5Mn, Mg-10Gd-0.5Mn, Mg-2Gd-1Mn, Mg-5Gd-1Mn, and Mg-10Gd-1Mn. The study examined the effects of varying extrusion rates (0.5, 1, and 2 mm/s) and extrusion temperatures (350 and 450 $\mathrm{^\circ\mathrm{C}}$) on microstructure, texture properties mechanical performance of these alloys \citep{cryst12081036}. 

\subsection{Microstructure binarization and statistics}
\label{sec3_binarization}

The OM data presented in \tabref{tab:matdata} includes the following conditions: AZ31 with an extrusion speed ($\mathbf{v_{ext}}$) of 0.6 mm/s and an extrusion temperature ($\mathbf{T_{ext}}$) of 500°C; ME21 with $\mathbf{v_{ext}}$ = 5.5 mm/s and $\mathbf{T_{ext}}$ = 450°C; and Mg-5Gd-1Mn with $\mathbf{v_{ext}}$ = 1 mm/s and $\mathbf{T_{ext}}$ = 450°C. These data are illustrated at scales of $200$, $100$, and $100 \,\upmu \text{m}$, respectively, in \figref{fig:OM_data}. Upon initial inspection, it is evident that the images contain a high density of information, including a significant amount of extraneous data. To address this, it is crucial to first maintain a uniform scale to achieve a coherent image dataset for microstructure analysis, ensuring a consistent micrometer-to-pixel ratio ($\upmu \text{m}/px$). Currently, for example, \figref{fig:OM_data_a} and \figref{fig:OM_data_b} have different $\upmu \text{m}/px$ ratios. Therefore, we need to crop tiles from images according to these ratios to create uniformly scaled microstructure images. For this purpose, tiles will be cropped at sizes of $90 \times 90$, $100 \times 100$, $120 \times 120$, and $150 \times 150 \,\upmu \text{m}$ and resized to $256 \times 256$ pixels. These four OM microstructure image datasets will also be compared against each other to understand the effect of scale selection on the mechanical property prediction for the extruded Mg-alloy dataset.

\begin{figure}[h]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/AZ31_extruded_500_0.6_0.17.JPG}
        \caption{}
        \label{fig:OM_data_a}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/ME21_extruded_450_5.5_0.107.jpg}
        \caption{}
        \label{fig:OM_data_b}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Mg_5Gd_1Mn_450_1_0.107.jpg}
        \caption{}
        \label{fig:OM_data_c}
    \end{subfigure}
    \caption{Sample microstructure images from the dataset in \tabref{tab:matdata}, generated via Light Optical Microscopy. (a) AZ31, $\mathbf{v_{ext}} = 0.6 \, \text{mm/s}$, $\mathbf{T_{ext}}$ = 500 $\mathrm{^\circ\mathrm{C}}$, scale = $200 \,\upmu \text{m}$, $\,\upmu \text{m}/px = 0.178$. (b) ME21, $\mathbf{v_{ext}} = 5,5 \, \text{mm/s}$, $\mathbf{T_{ext}}$ = 450 $\mathrm{^\circ\mathrm{C}}$, scale = $100 \,\upmu \text{m}$, $\,\upmu \text{m}/px = 0.107$. (c) Mg-5Gd-1Mn, $\mathbf{v_{ext}} = 1 \, \text{mm/s}$, $\mathbf{T_{ext}}$ = 450 $\mathrm{^\circ\mathrm{C}}$, scale = $100 \,\upmu \text{m}$, $\,\upmu \text{m}/px = 0.107$. }
    \label{fig:OM_data}
\end{figure}

To further reduce the information in these uniformly scaled images and enhance consistency, it is necessary to extract grain boundaries and inclusions. We will adapt and employ previously published image processing and machine learning techniques that have been successfully applied to OM-based microstructure images \citep{molecules27154826, NA2023113410}. Both of these approaches utilize pixel value-based image thresholding to enhance contrast, edge detection for grain boundary segmentation, CNN-based grain segmentation and various other image processing operations. These are employed in a specific order and combination to generate binary images, where pixel values of 1 represent grain boundaries and 0 represent all other areas. These methods also incorporate statistical analysis of the segmented grains to generate grain statistics, such as aspect ratio and equivalent diameter for each pixel. The microstructure binarization approach used in this research is illustrated in \figref{fig:binarization}. This approach involves two separate workflows: one for segmenting and binarizing the grain boundaries, including calculating grain size and aspect ratio statistics, and another for segmenting out the inclusions (black blobs in \figref{fig:OM_data}, which may be non-metallic particles or other alloy phases). The final binary microstructure image is obtained by combining these two segmentations.

\begin{figure*}[ht]
    \centering
    \includegraphics[width=0.8\textwidth]{Figures/Microstructure_Binarization.png}
    \caption{Microstructure Binarization workflow with calculation of grain statistics via grain boundary segmentation (blue) and inclusion segmentation (red).}
    \label{fig:binarization}
\end{figure*}

In the first workflow, the uniformly scaled microstructure image is initially denoised to eliminate any background jittering of the pixels. This is achieved using an image processing technique called non-local means (NLM) denoising. NLM denoising replaces each pixel’s value with a weighted average of similar pixels from across the entire image, not just nearby ones. This technique was introduced by \citep{1467423} in their 2005 paper “A Non-Local Algorithm for Image Denoising”. The denoised value at pixel $p$, 

\begin{equation}
u(p) = \frac{1}{C(p)} \int_{\Omega} v(q) f(p,q) \, \mathrm{d}q,
\end{equation}

where $ v(q) $ is the original pixel value, $f(p,q)$ is a similarity weight function, and $C(p)$ is a normalizing factor.

Contrast Limited Adaptive Histogram Equalization (CLAHE) is subsequently applied to enhance the contrast of the images while preventing noise amplification in homogeneous areas. This step is crucial for the subsequent grain segmentation process, as it helps to clearly distinguish between grains and grain boundaries. Unlike standard histogram equalization, CLAHE restricts the amplification of noise by clipping the histogram at a predefined threshold at a clip limit. The CLAHE thresholding function at pixel $p$, 

\begin{equation}
    T(I(p)) = \text{CDF}(I(p)) \cdot \text{clip limit},
\end{equation}

where $\text{CDF}(I(p))$ is the cumulative distribution function calculated from the histogram of the local tile and $\text{clip limit}$ is typically the maximum pixel value (e.g., 255 for an 8-bit image) \citep{Zuiderveld}. This method is particularly effective for OM imaging due to varying lighting conditions, as it improves visibility without over-amplifying noise.



\begin{figure}[h]
    \centering
    % Column captions
    \begin{minipage}{0.19\textwidth}
        \centering
        \caption*{CLAHE}
        \vspace{-0.5em} % Adjust this value as needed
    \end{minipage}
    \hfill
    \begin{minipage}{0.19\textwidth}
        \centering
        \caption*{SAM}
        \vspace{-0.5em} % Adjust this value as needed
    \end{minipage}
    \hfill
    \begin{minipage}{0.19\textwidth}
        \centering
        \caption*{Canny ED}
        \vspace{-0.5em} % Adjust this value as needed
    \end{minipage}
    \hfill
    \begin{minipage}{0.19\textwidth}
        \centering
        \caption*{Inclusions}
        \vspace{-0.5em} % Adjust this value as needed
    \end{minipage}
    \hfill
    \begin{minipage}{0.19\textwidth}
        \centering
        \caption*{Final}
        \vspace{-0.5em} % Adjust this value as needed
    \end{minipage}
    
    % Group 1
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/AZ31_extruded_500_0.6_0.17_extra_150_3_CLAHE_denoised.png}
        \label{fig:group1_fig1}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/AZ31_extruded_500_0.6_0.17_extra_150_3_segmented.png}
        \label{fig:group1_fig2}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/AZ31_extruded_500_0.6_0.17_extra_150_3_binary.png}
        \label{fig:group1_fig3}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/AZ31_extruded_500_0.6_0.17_extra_150_3_inclusion.png}
        \label{fig:group1_fig4}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/AZ31_extruded_500_0.6_0.17_extra_150_3_final.png}
        \label{fig:group1_fig5}
    \end{subfigure}
    \vspace{-1.6em} % Adjust this value as needed
    \caption*{(a) AZ31, $\mathbf{v_{ext}} = 0.6 \, \text{mm/s}$, $\mathbf{T_{ext}} = 500 \mathrm{^\circ\mathrm{C}}$, $x_\text{scale} = y_\text{scale} = 150 \,\upmu \text{m}$.}
    
    % Group 2
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/ME21_extruded_450_5.5_0.107_120_1_CLAHE_denoised.png}
        \label{fig:group2_fig1}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/ME21_extruded_450_5.5_0.107_120_1_segmented.png}
        \label{fig:group2_fig2}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/ME21_extruded_450_5.5_0.107_120_1_binary.png}
        \label{fig:group2_fig3}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/ME21_extruded_450_5.5_0.107_120_1_inclusion.png}
        \label{fig:group2_fig4}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/ME21_extruded_450_5.5_0.107_120_1_final.png}
        \label{fig:group2_fig5}
    \end{subfigure}
    \vspace{-1.6em} % Adjust this value as needed
    \caption*{(b) ME21, $\mathbf{v_{ext}} = 5.5 \, \text{mm/s}$, $\mathbf{T_{ext}} = 450 \mathrm{^\circ\mathrm{C}}$, $x_\text{scale} = y_\text{scale} = 120 \,\upmu \text{m}$.}
    
    % Group 3
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Mg_5Gd_1Mn_extruded_450_1_0.107_extra_120_0_CLAHE_denoised.png}
        \label{fig:group3_fig1}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Mg_5Gd_1Mn_extruded_450_1_0.107_extra_120_0_segmented.png}
        \label{fig:group3_fig2}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Mg_5Gd_1Mn_extruded_450_1_0.107_extra_120_0_binary.png}
        \label{fig:group3_fig3}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Mg_5Gd_1Mn_extruded_450_1_0.107_extra_120_0_inclusion.png}
        \label{fig:group3_fig4}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.19\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Mg_5Gd_1Mn_extruded_450_1_0.107_extra_120_0_final.png}
        \label{fig:group3_fig5}
    \end{subfigure}
    \vspace{-1.6em} % Adjust this value as needed
    \caption*{(c) Mg-5Gd-1Mn, $\mathbf{v_{ext}} = 1\, \text{mm/s}$, $\mathbf{T_{ext}} = 450\mathrm{^\circ\mathrm{C}}$, $x_\text{scale} = y_\text{scale} = 120 \,\upmu \text{m}$.}
    
    \caption{Three examples of Mg-alloy sample images at different stages of the microstructure binarization workflow.}
    \label{fig:binarization_photos}
\end{figure}


The zero-shot capability of the Segment Anything Model (SAM) from Meta is utilized to segment grain masks from the high-contrast microstructure images. SAM automatically adapts to different imaging conditions without requiring any labeled training data \citep{kirillov2023segment}. This makes SAM particularly suitable for microstructure grain segmentation in our dataset, as it effectively handles images with varied grain topology and lighting conditions. An implementation of the Segment Anything Model (SAM) is used to generate grain masks from the thresholded images, as illustrated by the three examples in \figref{fig:binarization_photos}. SAM creates a mask for each grain, and these masks are then overlaid to produce an image with all segmented grains, each colored differently. The area and bounding box of each mask are used to calculate statistics for each micrograph, specifically the grain size distribution, which includes the equivalent diameter and aspect ratio distribution of the grains. These statistics are presented in the form of binned histograms and are used as input for the final structure-property (S-P) model. The values for three different specimen micrograph at scale $150 \times 150 \, \upmu \text{m}$, are compared in binned histograms, as depicted in \figref{fig:grain_stats}. A template of 30 bins ranging from $5-200 \, \upmu \text{m}$ for grain boundary distribution and 3 bins ranging from $0-5$ for aspect ratio distribution is used to maintain data coherence. This approach facilitates the S-P model's ability to learn and recognize patterns effectively.


\begin{figure}[h]
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/distribution_eq_diameter_comparison_publication.pdf}
        \caption{}
        \label{fig:OM_data_a}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/distribution_aspect_ratio_comparison_publication.pdf}
        \caption{}
        \label{fig:OM_data_b}
    \end{subfigure}
    \caption{Microstructure statistics for scale = 150 $\upmu \text{m}$. (a) Grain size distribution and (b) Aspect ratio distribution for AZ31 ($\mathbf{v_{ext}} = 0.6 \, \text{mm/s}$, $\mathbf{T_{ext}}$ = 500 $\mathrm{^\circ\mathrm{C}}$), ME21 ($\mathbf{v_{ext}} = 5,5 \, \text{mm/s}$, $\mathbf{T_{ext}}$ = 350 $\mathrm{^\circ\mathrm{C}}$) and Mg-10Gd ($\mathbf{v_{ext}} = 1 \, \text{mm/s}$, $\mathbf{T_{ext}}$ = 450 $\mathrm{^\circ\mathrm{C}}$). }
    \label{fig:grain_stats}
\end{figure}


Next, to generate the binarized grain boundary image from the overlaid mask image, a multi-stage algorithm called Canny edge detection is used to effectively identify the edges of the grain masks. The algorithm computes the intensity gradients of the smoothed image using derivative filters (such as Sobel filters). This results in two gradient components, ( $G_x$ ) and ( $G_y$ ), which represent changes in intensity in the horizontal and vertical directions, respectively. The magnitude of the gradient is given by:

\begin{equation} 
    G = \sqrt{G_x^2 + G_y^2}, 
\end{equation}

which signifies the strength of the edge, while the direction is given by:

\begin{equation} 
    \theta = \tan^{-1}\left(\frac{G_y}{G_x}\right). 
\end{equation}

The next step is Non-Maximum Suppression, which thins out the edges by retaining only local maxima in the gradient direction, effectively suppressing non-edge pixels. Finally, two thresholds are applied to determine which edges are significant. Pixels with gradient magnitudes above a high threshold are marked as edges, while those below a low threshold are discarded \citep{Wang2009AnIC}.

Finally, morphological operations such as erosion and dilation of pixel boundaries are applied to the detected edges at the end of the binarization process. These operations help to form a clean, binarized grain boundary image (as seen in Canny ED images from \figref{fig:binarization_photos}), which can then be used to derive statistical descriptors.

In the second workflow (red workflow from \figref{fig:binarization}) for inclusion segmentation, the mean pixel value $\bar{I}_{\text{pixel}}$ and the standard deviation of the pixel value $\sigma_{I_{\text{pixel}}}$ are calculated for the cropped, uniformly scaled micrograph. These values are then used for adaptive thresholding to create binary image of inclusion. Adaptive thresholding is a technique in image processing that segments an image by dynamically determining the threshold value for each pixel based on the local characteristics of its neighborhood. The algorithm examines a small region around each pixel to compute a local threshold. The local threshold $ T $ can be calculated using:

\begin{equation} 
    T = \text{mean}(I_L) - C,
\end{equation}

where $ \text{mean}(I_L) $ is the mean intensity of the local neighborhood around pixel $ p $ and $ C $ is a constant, which is determined differently for each micrograph using $\bar{I}_{\text{pixel}}$ and $\sigma_{I_{\text{pixel}}}$.

To identify inclusions (which appear as black blobs) in the thresholded image, we use the `findContours` function from the OpenCV library. This powerful tool detects contours of pixel islands in images. Contours are simply curves that join all continuous points along a boundary with the same intensity. These contours are stored in a hierarchy based on their detection and retrieval order. Once detected, the contours and their hierarchy are filtered based on the average pixel value (intensity) of each contour being greater than 200 and the area of the contour being greater than 5. Additionally, any contours touching the rectangular boundary of the image are excluded from the final selection. All the contours are then drawn on a black background and filled with pixels. To obtain clean binary inclusions, further morphological operations (such as erosion and dilation) are performed, as shown in \figref{fig:binarization_photos} under “Inclusions.” The two binary images (one with grain boundaries and the other with inclusions) are then overlaid to create the finalized binary microstructure image. This image will be used to derive statistical descriptors of grain boundary features and inclusions, which will, in turn, be utilized by the structure-property (S-P) model to make property predictions.


\subsection{Uni-axial tensile property analysis}
\label{sec3_property}

Uni-axial tensile test evaluates material's ability to resist deformation under tension, providing crucial data such as yield strength ($\mathbf{\sigma_{y}}$), ultimate tensile strength ($\mathbf{\sigma_{max}}$), strain hardening exponent ($\mathbf{n}$) \citep{cryst11101193}, uniform strain ($\mathbf{\varepsilon_{pu}}$) and fracture strain ($\mathbf{\varepsilon_{pf}}$). Typical values for these properties for various Mg-alloys are presented in \tabref{tab:matdata}. These values are derived from the aforementioned studies where tensile tests were conducted on cylindrical rod samples with a diameter of 5 mm and a gauge length of 30 mm for the Z1, ZNd10, ZX10, Mg-Gd, and Mg-Gd-Mn alloys. For the AZ31, AZ31(HT10min450), ME21, ZNd10(HT10min450), and ZX10(HT10min450) alloys, flat specimens with a width of 40 mm and a thickness of 2 mm were used. All tests were performed at room temperature along the extrusion direction using a 50 kN testing machine (Zwick Z050) with a constant strain rate of $10^{-3} \, \text{s}^{-1}$.

The data obtained from these studies are presented in the form of engineering stress vs. strain tables, with yield point values assigned to each table. These yield point values are used to calculate the $\mathbf{\sigma_{y}}$. The maximum stress value in the tabular data is considered as $\mathbf{\sigma_{max}}$, and the corresponding plastic strain value is taken as the uniform strain. The plastic strain experienced by the material at the point of fracture is called fracture strain. It is determined for each table by identifying the point where either the data ends or the stress value starts to drop rapidly for each strain step. To characterize the plastic behavior of a material in the strain hardening region, we use the relation between true stress ($\mathbf{\sigma_{t}}$) and true strain ($\mathbf{\varepsilon_{t}}$),

\begin{equation}
    \mathbf{\sigma_{t}} = \mathbf{K} \mathbf{\varepsilon_{t}}^\mathbf{n},
\label{eq:eq_strain_hardeing_law}
\end{equation}

where $ K $ is the strength coefficient, which represents the stress level at a true strain of $\mathbf{\varepsilon_{t}} = 1 $. $\mathbf{n}$ is the strain hardening exponent, which indicates the rate of hardening of the material; it reflects how quickly the material strengthens with increasing strain in the plastic region, it begins at $\mathbf{\sigma_{y}}$ and extends to $\mathbf{\sigma_{max}}$. The corresponding indices for these points are identified from the stress-strain data. Subsequently, engineering stress and strain values within the plastic region are converted into true stress ($\mathbf{\sigma_{t}}$) and true strain ($\mathbf{\varepsilon_{t}}$) using, 

\begin{align}
    \mathbf{\sigma_{t}} &= \mathbf{\sigma_{e}} (1 + \mathbf{\varepsilon_{e}}), \\
    \mathbf{\varepsilon_{t}} &= \ln(1 + \mathbf{\varepsilon_{e}}),
\end{align}



where  $\mathbf{\sigma_{e}}$ and $ \mathbf{\varepsilon_{e}}$ are the engineering stress and strain, respectively. To facilitate the calculation of $\mathbf{n}$ and $\mathbf{K}$, these true stress and true strain values are then transformed into their logarithmic forms, yielding the expression for Eq.~\eqref{eq:eq_strain_hardeing_law} as,

\begin{equation}
    \ln(\mathbf{\sigma_{t}}) = \mathbf{n} \ln(\varepsilon_{t}) + \ln(\mathbf{K}).
\end{equation}

A linear regression analysis is then conducted on the log-transformed data to determine the slope and intercept of the resulting line. The slope of this regression line corresponds to $\mathbf{n}$, while the intercept provides the natural logarithm of $\mathbf{K}$. This approach allows for the characterization of material hardening behavior in the plastic deformation region of the stress-strain curve.



\section{Microstructure characterization}
\label{sec4}

In this section, we will explain the statistical descriptors used to characterize the binarized microstructure images, along with the libraries and implementations employed to calculate them. We will discuss the n-point spatial correlation function, which captures the spatial correlation within the microstructure images, and the gram matrices, which extract feature information such as grain boundary texture and grain boundary patterns from the $256 \times 256$ pixel images across four scales.

\subsection{N-point spatial correlation function}
The n-point spatial correlation function is a statistical tool used to describe the spatial arrangement and relationships within spatially distributed set of points of a material’s microstructure. This function quantifies how grain boundary features in a microstructure is positioned relative to each other at different distances. The term “n-point” refers to the number of points considered simultaneously, with higher values of n capturing more complex spatial relationships. The most common form for heterogeneous media is the 2-point function, which considers pairs of points and provides a measure of how the probability of finding a specific feature at one point is related to finding the same or another feature at a given distance and direction from it. The 2-point spatial correlation of a microstructure image $m[s, l]$ with $s$ pixels and $l$ phases (0 or 1 in our case due to the binary microstructure) can be represented as the conditional probability of finding local states,

\begin{equation} 
f_2 = f\left[r \mid l, l^{\prime}\right]=\frac{1}{S} \sum_s m[s, l] m\left[s+r, l^{\prime}\right], 
\label{eq:eq_2_point}
\end{equation}

where $S$ is the total number of pixels, $l$ and $l^{\prime}$ are the local states at two ends of a distance and orientation defined by vector $r$ \citep{Brough2016}. The implementation of 2-point statistics by \citep{Brough2016} in the Python library PyMKS (Materials Knowledge Systems in Python), an open-source framework designed for materials data science, is utilized for our purpose. The final dimension of the calculated 2-point spatial descriptor for each image is ($256 \times 256 \times 2$).

By examining triplets (3 points), one can capture more complex spatial patterns. This third order correlations are more sensitive to the exact geometrical arrangements in the material and can reveal intricate details about the grain boundary structure. The 3-point spatial correlation function extends the concept of the 2-point function by considering triplets of points. This function provides a measure of how the probability of finding a specific feature at one point is related to finding the same or another feature at two other points, defined by vectors $r_1$ and $r_2$,

\begin{equation} 
    f_3 = f\left[r_1, r_2 \mid l, l^{\prime}, l^{\prime\prime}\right]=\frac{1}{S} \sum_s m[s, l] m\left[s+r_1, l^{\prime}\right] m\left[s+r_2, l^{\prime\prime}\right], 
\label{eq:eq_3_point}
\end{equation}

where $S$ is the total number of pixels, and $l$, $ l^{\prime}$, and $l^{\prime\prime}$ are the local states at the distances and orientations defined by $r_1$ and $r_2$. This function captures more complex spatial relationships and provides detailed information about the geometrical arrangements and interactions within the microstructure. For the purpose of this study, the implementation of 2-point statistics from PyMKS is extended to 3-point statistics, as represented by Eq.~\eqref{eq:eq_3_point}. The final dimension of the calculated 3-point spatial descriptor for each image is ($256 \times 256 \times 3$).

\subsection{Gram matrices representations}

Gram matrices are a more recent, compact and powerful statistical descriptor used for microstructure images. They provide a compact representation of spatial relationships among points of grain boundary features, making them particularly useful for analyzing grain boundary structures and texture. \citep{GatysEB15} developed a robust algorithm for the statistical description of input image, where internal activations in convolutional layers of a pre-trained CNN is denoted as $F^{l}_{ij}$, where $l$ is the layer index, $i$ is the feature index, and $j$ is the pixel index. At each layer, the gram matrices can be captured between the feature index $i$ and feature index $k$:

\begin{equation} 
    G^{l}_{ik} = \sum_j F^{l}_{ij} F^{l}_{kj}, 
\end{equation}

where the summation represents the transpose of the two feature vectors, which encodes invariance to translations in the image, up to its boundaries \citep{Lubbers}. Compared to just the feature vector (activation filters from CNN), gram matrices provide a much richer statistical description. In this work, the implementation of gram matrices by \citep{seibert2022microstructure} is utilized for the binarized microstructure. This approach uses a pre-trained, normalized version of a 16-layer VGG network, based on the ResNet architecture. Activation filters from layers l = 2, 4, 8, 12 and 16 are extracted, and gram matrices are calculated for each layer. To form the final texture vector, all the gram matrices are normalized and concatenated as: $\hat{G}^l=\left(w_l / A_l\right) G^l$, to get a scaled texture vector, $\hat{G} = \left(\hat{G}^2, \hat{G}^4, \cdots\right)$. For a binary image of size ( $256 \times 256$ ), the calculated gram matrices dimensions are $(64, 64)$, $(128, 128)$, $(512, 512)$  and $(512,512)$.

\section{Texture characterization}
\label{sec5}

The texture data extracted from the experiments presented in \tabref{tab:matdata} is in the form of pole figures generated by XRD experiments, as shown in \figref{fig:PF}. The material data comprises extruded flat bands and extruded cylindrical bars. For flat bands, the preparation and measurement of the specimen are done on the normal plane, whereas for round bars, the specimen is prepared and measured in XRD on the cross-sectional plane. This creates a difference in the texture description in the data. Therefore, rotations in the Bunge Euler angles are performed using an open-source MATLAB toolbox for crystallographic texture analysis, MTEX \citep{BACHMANN20111720}. The rotation of the pole figure is done using the built-in functions within MTEX to maintain a uniform description of texture in the pole figure, as shown in \figref{fig:PF}. Sample symmetry is explained in this figure, the extrusion direction (ED) is oriented to the right (x-axis), the normal or second transverse direction for round bars is coming out of the plane (z-axis), and the other transverse direction is upwards (y-axis). Further, the uniform description of pole figures is used to calculate the ODF (see \figref{fig:ODF}), which describes the probability denisty of finding grains with crystal orientations represented by Bunge Euler angles: $\left\{ g = \left( \varphi_1, \Phi, \varphi_2 \right) \, \middle| \, 0 \leq \varphi_1 < 2\pi, \ 0 \leq \Phi \leq \pi/2, \ 0 \leq \varphi_2 < \pi/6 \right\}$ for Mg-alloys.

\begin{figure}[h]
    \centering
    \begin{subfigure}[b]{0.47\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Z1_300_2_pf.png}
        \caption{}
        \label{fig:PF}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.52\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/Z1_300_2_3d_odf.pdf}
        \caption{}
        \label{fig:ODF}
    \end{subfigure}
    \caption{Material texture characterization for Z1 ($\mathbf{v_{ext}} = 2 \, \text{mm/s}$, $\mathbf{T_{ext}}$ = 300 $\mathrm{^\circ\mathrm{C}}$). (a) Pole figure example from the dataset for texture on 1010 (left) and 0002 (right) planes with extrusion direction (ED), normal direction (ND) and transverse direction (TD) annotated. (b) 3D representtion of Orientation distribution function (ODF) for the corresponding pole figure. }
    \label{fig:texture}
\end{figure}

Because of the discrete description of the ODF in 5-degree increments, the ODF becomes high-dimensional data, making it challenging to use directly for Structure-Property (S-P) modeling. Therefore, the ODF can be analytically reduced into the expansion coefficients of the Generalized Spherical Harmonics (GSH) series. The harmonic series can be further reduced by leveraging the hexagonally close-packed (HCP) crystal symmetry of the Mg-alloy and the sample symmetry (orthotropic). This reduced representation of ODF, utilized in our study and also described by \citep{BUNGE19811},

\begin{equation} 
    f(g)=\sum_{l=0}^{\infty} \sum_{m=1}^{M(l)} \sum_{n=1}^{N(l)} F_l^{m n} \overset{\cdot}{\ddot{T}}_l^{m n}(g),
\end{equation}

where $F_l^{m n}$ are the expansion coefficients and $\overset{\cdot}{\ddot{T}}_l^{m n}(g)$ is the GSH basis function, reduced for orthotropic sample symmetry and HCP crystal symmetry. The combination of numbers $l$, $m$ and $n$ defines the number of dimensions that the series will be expanded for. Each combination corresponds to one expansion coefficient $F_l^{m n}$, and the symmetry constraints up to the chosen number for the expansion determine the dimensionality of the space \citep{BUNGE19811}. For our purpose, and most purposes, $l$ is chosen to be 16. This value is the default for orientation imaging microscopy analysis software, TexSEM Laboratories (TSL OIM).

Expansion coefficient pertaining to each orientation $\left( \varphi_1, \Phi, \varphi_2 \right)$ in the ODF denoted by $g_k$, can be represented as

\begin{equation}
{ }^k F_l^{m n}=(2 l+1) \overset{\cdot}{\ddot{T}} *{ }_l^{m n}\left(g_k\right) ,
\end{equation}

where $\overset{\cdot}{\ddot{T}} *{ }_l^{m n}$ is the complex conjugated reduced basis function. To calculate the expansion coefficients for the complete ODF, represented by a total of $N_{\textbf{crys}}$ orientations, the volume average of each individual expansion coefficient is computed,

\begin{equation}
\bar{F}_l^{m n}=\sum_{k=1}^{N_{\textbf{crys}}} { }^k \alpha^k F_l^{m n}, \sum_{k=1}^{N_{\textbf{crys}}} { }^k \alpha=1,0<{ }^k \alpha<1 ,
\end{equation}

where ${ }^k \alpha$ represents the weighting factor for each orientation $k$ \citep{BARRETT2019100328}. Implementing the above set of equations for the ODF in our dataset yields 551 coefficients, each with a real and imaginary value. These coefficients are vertically stacked to form a 1D vector of dimension 1102. 

\section{Dimensionality reduction techniques}
\label{sec6}

In this section, we will discuss the machine learning-based techniques used for reducing the dimensionality of the microstructure descriptors (discussed in Section \ref{sec4}) and texture descriptors (discussed in Section \ref{sec5}). The techniques that will be discussed include Principal Component Analysis (PCA), Isomap manifold learning, and Autoencoder-based vector embeddings. The discussion will also encompass which technique is used for which descriptor and the suitability of each technique for the respective descriptors.


\subsection{Principal component analysis (PCA)}
Principal Component Analysis (PCA) is a widely used dimensionality reduction technique. It centers the feature matrix of the descriptors around the mean by subtracting the average of all data points and then transforms it into a set of orthogonal components, capturing the significant variance in the data. The PCA representation for any descriptor replaces as

\begin{equation}
f_r=\sum_{k=1}^{R} \alpha_k \phi_{k r}+\bar{f}_r,
\end{equation}

where R is the total number of principal components, $\bar{f}_r$ is the ensemble mean for the descriptor, $ \phi_{k r}$ are the orthogonal basis, and $\alpha_k$ are the weights computed by PCA for each corresponding component \citep{KALIDINDI2015111}.

\begin{figure}[ht]
    \centering
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/PCA_odf.pdf}
        \caption{}
        \label{fig:PCA_odf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/manifold_odf.pdf}
        \caption{}
        \label{fig:manifold_odf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/embedding_odf.pdf}
        \caption{}
        \label{fig:embedding_odf}
    \end{subfigure}
    \caption{First three components ($\phi_1$, $\phi_2$ and $\phi_3$) for microstructure and texture descriptors with reduced dimensions using techniques like (a) Principal Component Analysis (PCA) for 3-point spatial correlations (scale = $100 \times 100 \,\upmu \text{m}$), colored for strain hardening exponent ($\mathbf{n}$). (b) Isomap manifold learning for gram matrices (at scale = $100 \times 100 \,\upmu \text{m}$), colored for tensile yield stress ($\mathbf{\sigma_y}$). (c) Autoencoder based vector embeddings for orientation distribution function (ODF), colored for fracture strain ($\mathbf{\varepsilon_{pf}}$). }
    \label{fig:dimensionalty_reduction}
\end{figure}

Although PCA can handle some forms of non-linearity in the data, it is most effective when applied to data with linearity due to the linear nature of PCA. For this reason, PCA is commonly chosen to reduce the dimensions of 2-point and 3-point spatial correlation functions used for statistically describing the binarized microstructure. This is because of the inherent linearity in the description of these correlation functions, as shown in Eq.~\eqref{eq:eq_2_point} and \eqref{eq:eq_3_point} \citep{PAULSON2017428}. Scikit-learn, a widely used open-source machine learning library by \citep{pedregosa2011scikit}, is utilized for PCA to reduce the dimensions for $f\left[r \mid l, l^{\prime}\right]$ and $f\left[r_1, r_2 \mid l, l^{\prime}, l^{\prime\prime}\right]$ to 100 and 120 respectively. \figref{fig:PCA_odf} shows the 3D representation of the first three principal components ($\phi_1$, $\phi_2$ and $\phi_3$) for  $f\left[r_1, r_2 \mid l, l^{\prime}, l^{\prime\prime}\right]$ of binary microstructure images from the dataset with scale = $100 \times 100 \, \upmu \text{m}$. Each data point in the figure is colored according to the value of the plastic property, strain hardening exponent ($\mathbf{n}$). The color gradient in the figure represents variations in property values captured by the reduced dimensions. The figure reveals clusters of points with similar property values and a discernible distribution trend. However, it provides an incomplete representation, as it is based on a single microstructure descriptor and only the first three principal components of that descriptor. Consequently, it is beneficial to incorporate multiple microstructure descriptors, including texture descriptors, and utilize additional principal components for the reduced descriptors as input features to enhance property prediction accuracy

\subsection{Manifold learning (Isomap)}

Isomap, a type of manifold learning technique, is a non-linear dimensionality reduction method that extends the concept of Multidimensional Scaling (MDS) by incorporating geodesic distances between data points along the manifold of reduced dimensional space. For each data point in the feature matrix, its k-nearest neighbors are identified and represented as a graph where nodes are the data points and edges represent connections between the neighboring points. Shortest distances are further calculated between each pair of points in the graph, usually by algorithms like Dijkstra's, resulting in a distance matrix (D). Finally, classical MDS is used to embed these geodesic distances into lower-dimensional space by finding a configuration (Y) that minimizes the stress function:

\begin{equation} \Omega(y) = \left(\sum_{i=1}^N \sum_{j=1}^N \left(D_{i j}-\left|y_i-y_j\right|\right)^2\right)^{-\frac{1}{2}}, \end{equation}

where $D_{i j}$ is the matrix of geodesic distances, $\left|y_i-y_j\right|$ are the Euclidean distances, and $N$ is the total number of components \citep{Jiaoyun}.

Isomap is very effective for non-linear data that lies on a curved manifold, as it preserves the local structure much better than linear techniques like PCA. By preserving geodesic distances, Isomap maintains the global geometric relationships in the data, leading to a more faithful representation of the intrinsic data structure in the lower-dimensional embedding. Multidimensional scaling has been previously applied to features based on gram matrices due to the non-linearity in the descriptors \citep{Engel2011ASO}. Further graph-based approaches to implement MDS, like Isomap, are also used to achieve low-dimensional embeddings that leverage the properties of gram matrices \citep{Lezoray2012}. The implementation in Scikit-learn is again utilized to compute 200 low-dimensional embeddings of gram matrices-based texture vector ($\hat{G}^l$), for all the images in the dataset. \figref{fig:manifold_odf} shows the first three dimensions ($\phi_1$, $\phi_2$, and $\phi_3$) for the binary microstructure images from the dataset with a scale of $100 \times 100 \, \upmu \text{m}$. Each data point in the figure is colored according to the value of the tensile yield stress ($\mathbf{\sigma_y}$). Here, the clustering of the colors in one portion of the 3D space and the gradient of the cluster hints at some relationship, although not fully complete, but at least indicative, between the mechanical property and the gram matrices-based texture vector. Isomap is also used for the ODF due to the non-linearity in the probability density function used to represent the orientation of grains using Bunge-Euler angles (see \figref{fig:ODF}). This approach helps to observe the effect of this dimensionality reduction on texture descriptors and their impact on the performance of property prediction.

\subsection{Vector Embedding (Autoencoder)}

The ODF is defined in a multidimensional space with Euler angles as each dimension. The relationship between these orientations and the probability is inherently non-linear due to the periodic nature of angular measurements. For instance, the integration over angular parameters to obtain volume fractions involves non-linear transformations, which complicate the direct interpretation. Additionally, the reconstruction of the ODF from experimental pole figure data often involves approximations that introduce more non-linearity, which does not necessarily lie on a curved manifold, even in higher dimensions \citep{Aganj2010}. Understanding this non-linearity is essential for accurately predicting mechanical properties, especially anisotropic properties in polycrystalline metallic alloys. Therefore, we explore a contemporary dimensionality reduction technique that is particularly effective for processing volumetric data with complex non-linearity. This technique utilizes a 3D CNN Autoencoder to efficiently extract features from the ODF.

The 3D Convolutional Autoencoder comprises two main components: an encoder and a decoder. The encoder compresses the data into a lower-dimensional latent space by applying 3D convolutional layers, batch normalization layers, and max pooling layers. These layers capture the spatial hierarchies and feature patterns inherent in the three-dimensional input data, such as the ODF. Conversely, the decoder reconstructs the original input from the compressed representation using 3D transposed convolutional layers. This process involves 3D upsampling of the latent representation back to its original dimensions, aiming to minimize the reconstruction error. \tabref{tab:autoencoder} presents the Autoencoder architecture, implemented using an open-source deep learning library with a Python interface, Keras, and employed for the compact learning of the ODF. The first column in \tabref{tab:autoencoder} represents the operator and the activation layer applied after the operator. The input to the encoder has a shape of $72 \times 19 \times 12$, corresponding to the three Bunge-Euler angles and one channel, which is the probability density function. The autoencoder consists of five convolutional layers and a final fully connected linear layer, which encodes the data to learn a 100-dimensional latent feature representation of the ODF. A dropout ratio of 0.2 is used to prevent overfitting and encourage more generalized feature learning. The decoder architecture starts with a fully connected linear layer, followed by five 3D transposed convolutional layers with sigmoid activation functions. This setup allows for learning complex patterns in the data with smooth gradients, ultimately reconstructing the ODF. The mean squared error between the input and the output of the Autoencoder is minimized using the Adam optimizer, with an exponential decay learning rate scheduler starting at 0.0001. The model is trained for a maximum of 1000 epochs with a batch size of 8. An early stopping criterion is applied to halt training if the metrics do not improve for 5 consecutive epochs. \figref{fig:embedding_odf} illustrates the first three dimensions of the 100-dimensional features learned by the autoencoder for the 119 ODFs in our dataset. Each data point in the figure is colored according to the value of the fracture strain ($\mathbf{\varepsilon_{pf}}$). The color gradient indicates a relationship/trend between the lower-order representation of texture descriptors and mechanical property, $\mathbf{\varepsilon_{pf}}$.

\begin{table}[H]
\centering
\caption{\newline 3D Convolutional Autoencoder architecture for encoding Orientation Distribution Function (ODF).}
\label{tab:autoencoder}
\renewcommand{\arraystretch}{1.4}
\resizebox{\linewidth}{!}{
\begin{tabular}{ccccc} 
\hline
\multicolumn{5}{c}{\textbf{Encoder Architecture}} \\
\hline
Layer (Activation Function) & Kernel Size & Channels & Output Shape & \# of Param \\
\hline
Input Layer & - & 1 & $(72, 19, 12, 1)$ & 0 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 16 & $(72, 19, 12, 16)$ & 448 \\
Batch Normalization & - & - & $(72, 19, 12, 16)$ & 64 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 32 & $(72, 19, 12, 32)$ & 13,856 \\
Batch Normalization & - & - & $(72, 19, 12, 32)$ & 128 \\
Max-Pooling Layer & $2 \times 2 \times 2$ & - & $(36, 9, 6, 32)$ & 0 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 64 & $(36, 9, 6, 64)$ & 55,360 \\
Batch Normalization & - & - & $(36, 9, 6, 64)$ & 256 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 64 & $(36, 9, 6, 64)$ & 110,656 \\
Batch Normalization & - & - & $(36, 9, 6, 64)$ & 256 \\
Max-Pooling Layer & $2 \times 2 \times 2$ & - & $(18, 4, 3, 64)$ & 0 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 128 & $(18, 4, 3, 128)$ & 221,312 \\
Batch Normalization & - & - & $(18, 4, 3, 128)$ & 512 \\
Max-Pooling Layer & $2 \times 2 \times 2$ & - & $(9, 2, 1, 128)$ & 0 \\
Flatten Layer & - & - & $(2304)$ & 0 \\
Dropout Layer (0.2) & - & - & $(2304)$ & 0 \\
Dense Layer (Linear) & - & 100 & $(100)$ & 230,500 \\
\hline
\multicolumn{5}{c}{\textbf{Decoder Architecture}} \\
\hline
Layer (Activation Function) & Kernel Size & Channels & Output Shape & \# of Param \\
\hline
Dense Layer (ReLU) & - & 128 & $(4608)$ & 465,408 \\
Reshape Layer & - & - & $(9, 4, 6, 128)$ & 0 \\
Upsampling Layer & $2 \times 2 \times 2$ & - & $(18, 8, 12, 128)$ & 0 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 128 & $(18, 8, 12, 128)$ & 442,496 \\
Upsampling Layer & $2 \times 3 \times 1$ & - & $(36, 24, 12, 128)$ & 0 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 64 & $(36, 24, 12, 64)$ & 221,248 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 64 & $(36, 24, 12, 64)$ & 110,656 \\
Upsampling Layer & $2 \times 1 \times 1$ & - & $(72, 24, 12, 64)$ & 0 \\
Convolutional Layer (LeakyReLU) & $3 \times 3 \times 3$ & 32 & $(72, 24, 12, 32)$ & 55,360 \\
Convolutional Layer (LeakyReLU) & $1 \times 3 \times 1$ & 16 & $(72, 22, 12, 16)$ & 1,552 \\
Convolutional Layer (Sigmoid) & $1 \times 3 \times 1$ & 1 & $(72, 20, 12, 1)$ & 49 \\
Cropping Layer & - & - & $(72, 19, 12, 1)$ & 0 \\
\hline
\end{tabular}
}
\end{table}


\section{Lower-order modeling of Structure-Property linkages}
\label{sec7}

The reduced-order microstructure descriptors, including 2-point correlations, 3-point correlations, gram matrices, binned histogram representations of grain size distribution, and aspect ratio distribution, as well as reduced-order texture descriptors, are all shaped into a 1D vector. This serves as the input to the non-linear regression models discussed in this section for predicting mechanical properties. The descriptors will be investigated in various combinations to evaluate their performance in property prediction. For this purpose, non-linear, non-parametric regressors are chosen. Non-linear regressors are selected due to the complex patterns in the data that need to be linked to the corresponding mechanical properties. Non-parametric regressors are preferred because there are no predefined functional forms specified for the relationships between the structure and properties. The shape and complexity of the model should adapt to these linkages, making non-parametric regression models more suitable for this problem. Gaussian Process (GP) regression is a powerful non-parametric method used for modeling complex, non-linear relationships without assuming a specific functional form. The performance of the GP regressor is compared to that of a random forest and a Multi-Layer Perceptron (MLP) regressor. XGBoost, a non-parametric regressor, which is based on random forest and gradient boosting, is also evaluated. Although the MLP regressor is parametric due to the predetermined number of layers and neurons per layer, it remains a good choice for this application. This is because these numbers are considered hyper parameters and can be fine-tuned to accurately model the S-P relationship. For predicting each mechanical property a different regressor is trained with same inputs but different mechanical property as label.

\subsection{Random forest regression (XGBoost)}
XGBoost, or eXtreme Gradient Boosting, is a powerful machine learning algorithm that enhances the gradient boosting framework through piecewise constant approximations in decision trees. It is most effective when interactions between features and threshold values are important. The core of XGBoost's optimization lies in its objective function,

\begin{equation}
    L(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \Omega(f),
\end{equation}

where $l(y_i, \hat{y}_i)$ represents the loss function, which measures how well the predicted values $\hat{y}_i$ match the actual values $y_i$. XGBoost typically uses a squared loss function for regression tasks,

\begin{equation}
     l(y_i, \hat{y}_i) = (y_i - \hat{y}_i)^2,
\end{equation}

which quantifies the difference between actual and predicted values. $\Omega(f)$ is a regularization term that penalizes model complexity, helping to prevent overfitting by controlling the size of the trees in the ensemble. The regularization term can be defined as:

\begin{equation}
    \Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2,
\end{equation}

where $T$ is the number of leaves in the tree, $w_j$ are the weights of each leaf, $\gamma$ controls the minimum loss reduction required to make a further partition on a leaf node and $\lambda$ is a regularization parameter that penalizes large weights. XGBoost employs a gradient descent approach to minimize the objective function. At each iteration, it fits a new model (often a decision tree) to the negative gradient of the loss function \citep{Tianqi}. Hyperparameter selection is done using a 5-fold cross validation strategy over the training set. An early stopping criterion is applied for regularization, if the evaluation metric (mean squared error) does not improve for 10 consecutive iterations.

\subsection{Multi layer perceptron (MLP)}
An MLP regressor is a type of fully connected artificial neural network used for predicting continuous values. It consists of multiple layers of interconnected neurons, including an input layer, one or more hidden layers, and an output layer. Each neuron in the hidden layers computes a weighted sum of its inputs and applies an activation function $ h^{(\ell)}_i $, for the $ \ell $-th layer, the output of the $ i $-th neuron 

\begin{equation}
    h^{(\ell)}_i = \Psi^{(\ell)}\left(\sum_j w^{(\ell)}_{ij} h^{(\ell-1)}_j + b^{(\ell)}_i\right),
\end{equation}

where $ w^{(\ell)}_{ij} $ is the weight connecting neuron $ j $ from layer $ \ell-1 $ to neuron $ i $ in layer $ \ell $, $ b^{(\ell)}_i $ is the bias term for neuron $ i $ and $ \Psi^{(\ell)} $ is the activation function for layer $ \ell $. The activation function chosen for hidden layer of MLP regressor is Rectified Linear Unit (ReLU) which introduces non-linearity while being computationally efficient.

The final output of the MLP, which represents the predicted value,

\begin{equation}
    y = h^{(L)} = \Psi^{(L)}\left(\sum_j w^{(L)}_{j} h^{(L-1)}_{j} + b^{(L)}\right),
\end{equation}

where L denotes the output layer, which uses linear activation function for alloying continuous output values (important for regression tasks) for real numbers \citep{Bourlard1994}. MLP regressor is typically trained using back propagation along with Adam optimizer with the goal to minimize mean squared error loss function. The number of hidden layers, the number of neurons per layer, the learning rate, and the regularization parameter (alpha) are considered for hyperparameter optimization to enhance prediction accuracy. The training set is further divided into five folds, and a k-fold cross-validation strategy is employed for hyperparameter optimization.

\subsection{Gaussian process regression (GP)}

GP Regression utilizes the properties of Gaussian processes to make predictions about unknown functions.  In this context, a Gaussian process can be understood as a collection of random variables, any finite number of which have a joint Gaussian distribution. This characteristic allows GP regression to provide a flexible framework for modeling the underlying relation between co-variates and response. GP not only predicts the mean of the output variable but also provides a measure of uncertainty associated with these predictions \citep{Wang_2023}. This is achieved through the posterior distribution derived from the prior distribution and observed data. 

The GaussianProcessRegressor class from Scikit-learn is utilised because of its efficient implementation with support for various kernel functions and automatic hyperparameter optimization. In GP regression, the underlying function $f(\mathbf{x})$ is modeled as a realization of a Gaussian process,

\begin{equation}
    f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}')),
\end{equation}

where $ m(\mathbf{x}) $ is the mean function (often assumed to be zero), and $ k(\mathbf{x}, \mathbf{x}') $ is the covariance kernel function that encodes assumptions about the smoothness or periodicity of $ f(\mathbf{x}) $. Radial Basis Function (RBF) kernel,

\begin{equation}
     k_{\text{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2l^2}\right),
\end{equation}

where $l$ is the length scale parameter, is chosen as $ k(\mathbf{x}, \mathbf{x}') $. The observed data $ y_i $ at $ \mathbf{x}_i $ are assumed to be noisy realizations,

\begin{equation}
    y_i = f(\mathbf{x}_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_n^2).
\end{equation}

Given $ n $ training samples $ \mathcal{D} = \{ (\mathbf{x}_i, y_i) \}_{i=1}^n $, the predictive distribution at a new input $ \mathbf{x}_* $ is Gaussian with mean $ \mu(\mathbf{x}_*) $ and variance $ \sigma^2(\mathbf{x}_*) $,

\begin{equation}
    \mu(\mathbf{x}_*) = \mathbf{k}_*^\top \left( \mathbf{K} + \sigma_n^2 \mathbf{I} \right)^{-1} \mathbf{y},
\end{equation}

 \begin{equation}
     \sigma^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^\top \left( \mathbf{K} + \sigma_n^2 \mathbf{I} \right)^{-1} \mathbf{k}_*,
 \end{equation}

 where $ \mathbf{K} \in \mathbb{R}^{n \times n} $ is the covariance matrix with $ K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) $, $ \mathbf{k}_* \in \mathbb{R}^n $ is the covariance vector between $ \mathbf{x}_* $ and the training points, $ \sigma_n^2 $ is the noise variance, and $ \mathbf{y} \in \mathbb{R}^n $ is the vector of observed targets. Further, GaussianProcessRegressor optimizes the hyperparameter to train the model on the data by maximizing log-marginal likelihood,

\begin{equation}
     \log \left( p(\mathbf{y} \mid \mathbf{X}, \theta) \right) = -\frac{1}{2} \mathbf{y}^\top \left( \mathbf{K}_\theta + \sigma_n^2 \mathbf{I} \right)^{-1} \mathbf{y} - \frac{1}{2} \log \left( \det \left( \mathbf{K}_\theta + \sigma_n^2 \mathbf{I} \right) \right) - \frac{n}{2} \log 2\pi.
\end{equation}

Finally, predictions at test points are made by computing the joint distribution of observed outputs and outputs at new inputs. From this, the conditional distribution of $f(X_*)$ is computed, which ultimately yields mean predictions and uncertainty quantification \citep{Schreiter}. Since GP regression models are numerically similar to kernel regression, they benefit from an additional well-defined statistical framework. This framework enables the optimization of the likelihood function for estimating hyperparameters without the need for cross-validation.

\subsection{Shapley Additive Explanation Value (SHAP value)}
\label{sec7_4}

The non-linear regression models discussed above are often regarded as "black boxes" due to their ability to take in input arguments and return target output values without revealing the internal decision-making process. The complexity of these models increases with the intricacy of input features and their interdependencies during output value prediction, which in turn hampers their interpretability. Interpretability refers to the ability of a human to comprehend the impact of each input feature on the model's output. This becomes particularly challenging in this research due to the diverse range of input features used to predict mechanical properties. To address such complexities, the quantification of feature dependence or importance is crucial. In the field of machine learning, feature additive contribution models are employed to enhance the explainability of these complex models \citep{covert2020understandingglobalfeaturecontributions}. SHAP (SHapley Additive exPlanations) leverages the concept of game theory known as Shapley values, which provide a method for fairly distributing gains and losses among multiple contributors in a collaboration. In SHAP value analysis, the estimation of Shapley values is performed using a weighted linear regression with a kernel estimator \citep{lundberg2017unifiedapproachinterpretingmodel}. It is utilized in this research to understand the feature importance of microstructure statistics, microstructure, and texture descriptors in predicting mechanical properties.

The goal in SHAP value analysis for machine learning models is to fairly attribute the difference between the model's prediction for a given data point and the mean prediction to each feature. The value of feature ($i$) is calculated as,

\begin{equation}
    \text{SHAP}(x_i) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \, (|N| - |S| - 1)!}{|N|!} \Big[ f(S \cup \{i\}) - f(S) \Big],
    \label{eq:SHAP}
\end{equation}

where, $N$ is the set of all features, $S$ is a subset of features excluding $i$, $S \subseteq N \setminus \{i\}$. $f(S)$ is the model's prediction when only the features in subset $S$ are present. $|S|$ is the cardinality of subset $S$ and $|N|$ is the total number of features.

The marginal contribution of each feature $i$ is represented by the term $f(S \cup \{i\}) - f(S)$ ie. difference in the model's output when the feature is included in the subset $S$. The term before this in Eq.~\eqref{eq:SHAP}, ensures that the contributions are averaged across all possible subsets $S$ and summation at the end is done to attribute equal considerations to all feature combinations. Computation of SHAP values involves considering all $2^n$ subsets of features, making it computationally expensive for large $n$. Therefore, approximations such as Kernel and Tree explainer are used according to the nature of the regression model. For the purpose of this study, Tree explainer from the implementation done in the SHAP library by \citep{lundberg2019explainableaitreeslocal}, is used to understand the feature importance for GP, MLP and XGBoost regressors.


\section{Results and discussions}
\label{sec8}

\begin{table*}[ht]
\centering
\caption{\newline Input feature vectors and the sets of their various combinations which are used for investigating the performance of non-linear regression models in predicting mechanical properties, $\mathbf{\Omega}$ is used for regressor performance and length scale, $\mathbf{\Psi}$ is used for microstructure descriptors and $\mathbf{\Phi}$ is used for texture descriptors.}
\label{tab:feature}
\resizebox{\textwidth}{!}{
\begin{tblr}{
  row{2} = {c},
  row{4} = {c},
  row{5} = {c},
  row{7} = {c},
  row{8} = {c},
  row{10} = {c},
  cell{1}{1} = {r=2}{},
  cell{1}{2} = {r=2}{c},
  cell{1}{3} = {r=2}{c},
  cell{1}{4} = {r=2}{c},
  cell{1}{5} = {r=2}{c},
  cell{1}{6} = {c=3}{c},
  cell{3}{1} = {r=3}{},
  cell{3}{2} = {c},
  cell{3}{3} = {c},
  cell{3}{4} = {c},
  cell{3}{5} = {c},
  cell{3}{6} = {c},
  cell{3}{7} = {c},
  cell{3}{8} = {c},
  cell{6}{1} = {r=3}{},
  cell{6}{2} = {c},
  cell{6}{3} = {c},
  cell{6}{4} = {c},
  cell{6}{5} = {c},
  cell{6}{6} = {c},
  cell{6}{7} = {c},
  cell{6}{8} = {c},
  cell{9}{1} = {r=2}{},
  cell{9}{2} = {c},
  cell{9}{3} = {c},
  cell{9}{4} = {c},
  cell{9}{5} = {c},
  cell{9}{6} = {c},
  cell{9}{7} = {c},
  cell{9}{8} = {c},
  vline{2} = {3,4,5,6,7,8,9,10}{},
  hline{1,3,6,9,11,12} = {-}{},
  hline{2} = {6-9}{},
}
                                              & Input feature vectors        & {Dimensionality\\~reduction} & Size  & Symbol                                                        &Input sets for investigation\\
                                              &                                         &                              &       &                                                                && $\mathbf{\Omega}$   & $\mathbf{\Psi}$ & $\mathbf{\Phi}$ \\
\begin{sideways}Microstructure~\end{sideways} & 2-point spatial correlation function    & PCA                          & $100$ & $ ^\mathbf{PCA} f_2$                                           &&                     &          &          \\
                                              & 3-point spatial correlation function    & PCA                          & $120$ & $ ^\mathbf{PCA} f_3$                                           && \bigtick            &          & \bigtick \\
                                              & Gram Matrices based texture vector      & Isomap                       & $200$ & $ ^\mathbf{ISO} \hat{G}$                                       && \bigtick            &          & \bigtick \\
\begin{sideways}Texture\end{sideways}         & GSH coefficient vector                  & Isomap                       & $60$  & $ ^\mathbf{ISO} \bar{F}_l^{mn}$                                 && \bigtick            & \bigtick &          \\
                                              & Orientation Distribution Function (ODF) & Isomap                       & $60$  & $ ^\mathbf{ISO} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$ &&                     &          &          \\
                                              & Vector embeddings of ODF                & Autoencoder                  & $100$ & $ ^\mathbf{VEC} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$ &&                     &          &          \\
\begin{sideways}Stats.~\end{sideways}         & Aspect Ratio distribution               & None                         & $30$  & $\mathbf{AR}$                                                  && \bigtick            & \bigtick & \bigtick \\
                                              & Equivalent diameter distribution        & None                         & $30$  & $d_\mathbf{eq}$                                                && \bigtick            & \bigtick & \bigtick 
\end{tblr}
}
\end{table*}

In this section, we compare the prediction performance of the models discussed in Section \ref{sec7} using the input features outlined in Sections \ref{sec4}, \ref{sec5}, and \ref{sec6}. This comparison is conducted using parity plots, which compare true and predicted values, as well as metrics such as mean absolute error (MAE), mean absolute percentage error (MAPE), and the coefficient of determination ($\text{R}^2$). Among the mechanical properties discussed in Section \ref{sec3_property}, we focus the discussion on the prediction results for the strain hardening exponent, $\mathbf{n}$, and yield stress, $\mathbf{\sigma_{y}}$. The best performance for prediction of ultimate tensile stress, $\mathbf{\sigma_{max}}$, uniform strain, $\mathbf{\varepsilon_{pu}}$, and fracture strain $\mathbf{\varepsilon_{pf}}$ are further shown in \ref{app1}. In this section, we will discuss four different investigations conducted in this study to identify the best configuration for the modules of the pipeline mentioned in \figref{fig:paper_schematic} for accurate mechanical property prediction. \tabref{tab:feature} lists the input feature vectors and sets of these input feature vectors that will be used for these investigations. It also includes the symbol assigned to them, dimensionality reduction technique applied to them, their final reduced size, and whether they describe microstructure, microstructure statistics, or texture. 

\begin{itemize}
    \item In the first investigation, the suitability of the three regression models—GP, XGBoost, and MLP—to the dataset is evaluated. This is crucial to determine which model performs better than the others for the same input set, $\mathbf{\Omega}$.
    \item In the second investigation, the three reduced microstructure descriptors ($^\mathbf{PCA} f_2$, $^\mathbf{PCA} f_3$, and $^\mathbf{ISO} \hat{G}$) and their combinations are examined for their performance in predicting $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ using the input set $\mathbf{\Psi}$.
    \item In the third investigation, three different reduced-order texture descriptors ($^\mathbf{ISO} \bar{F}_l^{mn}$, $^\mathbf{VEC} g{\left( \varphi_1, \Phi, \varphi_2 \right)}$, and $ ^\mathbf{ISO} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$) are assessed for their performance in property prediction using the input set $\mathbf{\Phi}$. 
    \item Finally, the fourth investigation is conducted using the input set $\mathbf{\Omega}$ to understand the impact of length scale ($90$, $100$, $120$ and $150  \,\upmu \text{m}$) in reduced-order microstructure descriptors on property prediction accuracy.
\end{itemize}

These investigations are also further employed to understand the dependence of the target mechanical properties on the input microstructure and texture descriptors. This interpretation is further quantified using SHAP values, as presented in Section \ref{sec8_5}.

\subsection{Investigating non-linear regressors }
\label{sec8_1}

The best-performing configurations of hyperparameters for the GP, XGBoost, and MLP regressors are used to predict $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ for the input set $\mathbf{\Omega}$ at a scale of $100 \,\upmu \text{m}$. The fit of the model to the training and testing data for predicting $\mathbf{n}$ is illustrated using parity plots (true vs. predicted values) in \figref{fig:parity_regressor_strain}. It is observed that in all the parity plots, there are groups of points with the same true value but different predicted values. This occurs because tiling and multiple images taken from the same specimen might have different microstructure descriptors and microstructure statistics, yet share the same mechanical performance label. The grouping and alignment of these points also indicate that the models are able to learn some patterns in linking slightly different microstructure descriptors to the same output property label.

A similar comparison is observed between the models for predicting $\mathbf{\sigma_{y}}$ for the input set $\mathbf{\Omega}$ at a scale of $100 ,\upmu \text{m}$, as shown in \figref{fig:parity_regressor_yield}. The MLP regressor exhibits higher train MAPE values of 3.98\%, with train data points scattered further from the perfect agreement line in \figref{fig:yield_stress_MLP}, indicating underfitting and an inability to effectively learn the patterns in the data. This is further supported by high variance in the test set and a low $\text{R}^2$ value of 0.42. The GP regressor again shows overfitting compared to the XGBoost regressor, as evidenced by the difference in its train and test MAPE. Consequently, the model does not generalize well to the unseen test data, yielding a test MAPE of 13.15\%, which is the highest among the three regressors (see \figref{fig:mape_bar_yield_stress}). XGBoost once again performs best in terms of predicting the unseen data, with a test MAPE of 9.67\%. However, a few data points from a single material specimen perform considerably worse than others due to under representation in the train set (see \figref{fig:yield_stress_XGB}). Overall, the performance of all the regressors is not as good for predicting $\mathbf{\sigma_{y}}$ compared to $\mathbf{n}$ for the input set $\mathbf{\Omega}$ at a scale of $100 \,\upmu \text{m}$. Additionally, Appendix Section \ref{app2} summarizes all the performance metrics for different configurations of input features/sets across all three regressors.


\begin{figure}[t]
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_strain_hardening.pdf}
        \caption{}
        \label{fig:mape_bar_strain_hardening}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_yield_stress.pdf}
        \caption{}
        \label{fig:mape_bar_yield_stress}
    \end{subfigure}
    \caption{Overall performance comparison of Gaussian Process, Multi Layer Perceptron and XGBoost regressors using mean absolute percentage error (MAPE) bar graph for predicting (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, using the input set $\mathbf{\Omega}$, at scale = $100 \,\upmu \text{m}$.}
    \label{fig:bar_regressor}
\end{figure}


\begin{figure}[!t]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_GP_3point_gram_GSH.pdf}
        \caption{}
        \label{fig:strain_hardening_GP}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_3point_gram_GSH_regressor.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_MLP_3point_gram_GSH.pdf}
        \caption{}
        \label{fig:strain_hardening_MLP}
    \end{subfigure}
    \caption{True and predicted values of Strain Hardening Exponent, $\mathbf{n}$, for (a) Gaussian Process, (b) XGBoost and (c) Multi Layer Perceptron regressors using input feature set $\mathbf{\Omega}$, at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_regressor_strain}
\end{figure}

All GP, XGBoost, and MLP models fit well on the training data, with MAE of 0.007 and MAPE of 3.09\% for GP, train MAE of 0.004 and MAPE of 1.73\% for XGBoost, and train MAE of 0.01 and MAPE of 4.47\% for MLP. Nevertheless, the $\text{R}^2$ value of 0.95 and slightly worse training metrics indicate underfitting of the training data for the best-performing MLP model. This is also evident by examining the training data (blue) in \figref{fig:strain_hardening_MLP}, which is not the case with GP and XGBoost. Although the $\text{R}^2$ value of 0.97 is better for GP, it does not necessarily mean that it learns the inherent patterns of the material data better than MLP. For predicting the test data, GP performs worse than MLP, with MAE and MAPE of 0.026 and 10.60\% for GP, compared to MAE and MAPE of 0.018 and 8.6\% for MLP. The GP regressor comparatively overfits slightly more to the training data when compared to the XGBoost regressor, as indicated by the greater difference in Train and Test MAPE for GP.

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_GP_3point_gram_GSH.pdf}
        \caption{}
        \label{fig:yield_stress_GP}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_3point_gram_GSH_regressor.pdf}
        \caption{}
        \label{fig:yield_stress_XGB}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_MLP_3point_gram_GSH.pdf}
        \caption{}
        \label{fig:yield_stress_MLP}
    \end{subfigure}
    \caption{True and predicted values of Yield Stress, $\mathbf{\sigma_{y}}$, for (a) Gaussian Process, (b) XGBoost and (c) Multi Layer Perceptron regressors using input feature set $\mathbf{\Omega}$, at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_regressor_yield}
\end{figure}

\figref{fig:mape_bar_strain_hardening} compares the test and train predictions of all the models and shows that XGBoost performs superior to the other two models on the unseen test data, with a MAPE of 6.67\% in estimating the value of $\mathbf{n}$. This is also clear from \figref{fig:strain_hardening_XGB}, where the test data (orange points) are mostly very close to the perfect agreement line, apart from data from one or two material specimens, which reduced the test $\text{R}^2$ value to 0.72. This variance can be attributed to the fact that this material type from the test set does not have adequate representation in the train set, which can be improved by increasing the number of data points to enhance this representation.


\subsection{Investigating microstructure descriptors }

In Section \ref{sec8_1}, we found that XGBoost model learns and generalizes well over the data, as evidenced by its test lowest test and train MAPE in predicting both $\mathbf{n}$ and $\mathbf{\sigma_{y}}$. Therefore, XGBoost models will be used for the next three investigations. \figref{fig:mape_bar_strain_hardening_microstructure} shows the comparison of the XGBoost regressor in predicting $\mathbf{n}$ using $^\mathbf{PCA} f_2$, $^\mathbf{PCA} f_3$, $^\mathbf{ISO} \hat{G}$, and the combination of ($^\mathbf{PCA} f_3$, $^\mathbf{ISO} \hat{G}$) along with the input set $\mathbf{\Psi}$ at a scale of $100 ,\upmu \text{m}$. The PCA-reduced 3-point spatial correlation function performs the worst with a test MAPE of 9.5\%, while the isomap-reduced gram matrices, $^\mathbf{ISO} \hat{G}$, perform much better with a test MAPE of 6.97\%. This indicates that the style and texture of grain boundaries are more indicative of $\mathbf{n}$, making gram matrices a better descriptor of the microstructure features when directly compared to 2-point or 3-point spatial correlations for our curated dataset of extruded Mg-alloys. Although spatial correlation functions contains information of a spatial distribution of pixels with grain boundaries, which is absent in gram matrices, it is observed that the combination of ($^\mathbf{PCA} f_3$, $^\mathbf{ISO} \hat{G}$) performs slightly better than just $^\mathbf{ISO} \hat{G}$, with test MAPE of 6.67\%. The combination of ($^\mathbf{PCA} f_2$, $^\mathbf{ISO} \hat{G}$) was also tested but not included in the showcased results, as the improvement of the combination ($^\mathbf{PCA} f_3$, $^\mathbf{ISO} \hat{G}$) was more significant.


\begin{figure}[!t]
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_strain_hardening_microstructure.pdf}
        \caption{}
        \label{fig:mape_bar_strain_hardening_microstructure}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_yield_stress_microstructure.pdf}
        \caption{}
        \label{fig:mape_bar_yield_stress_microstructure}
    \end{subfigure}
    \caption{Overall performance comparison of different microstructure descriptor based input features using mean absolute percentage error (MAPE) bar graph for predicting (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, using the input set $\mathbf{\Psi}$, at scale = $100 \,\upmu \text{m}$ using XGBoost regressor.}
    \label{fig:bar_microstructure}
\end{figure}


\begin{figure}[!h]
    \centering
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_3point_gram_GSH.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_3point_gram_GSH}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_2point_GSH.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_3point_gram_GSH}
    \end{subfigure}
    \caption{True and predicted values of (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor using input features $ ^\mathbf{PCA} f_3$, $ ^\mathbf{ISO} \hat{G}$  and input set $\mathbf{\Psi}$, at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_microstructure}
\end{figure}

There is no significant difference in the prediction performance of $\mathbf{\sigma_{y}}$ using the microstructure descriptors—$^\mathbf{PCA} f_2$, $^\mathbf{PCA} f_3$, $^\mathbf{ISO} \hat{G}$, and the combination of ($^\mathbf{PCA} f_3$, $^\mathbf{ISO} \hat{G}$)—along with the input set $\mathbf{\Psi}$ at a scale of $100 ,\upmu \text{m}$ (see \figref{fig:mape_bar_yield_stress_microstructure}). The difference in test MAPE between the best-performing $^\mathbf{PCA} f_2$ and the worst-performing $^\mathbf{PCA} f_3$, $^\mathbf{ISO} \hat{G}$ is just 0.4\%. This indicates that contribution of microstructure descriptor in prediction of $\mathbf{\sigma_{y}}$ is limited. Additionally, all microstructure descriptors perform comparatively worse (test MAPE greater than 9.2\%), suggesting that the length scale of $100 \,\upmu \text{m}$ may not be suitable and may not contain enough information to learn the patterns or indicate $\mathbf{\sigma_{y}}$. The influence of length scales is further investigated in Section \ref{sec8_4}. The comparison of true vs. predicted values of $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ for the best-performing microstructure descriptors identified in this investigation is shown in \figref{fig:parity_microstructure}. The figure again demonstrates that the model is able to accurately learn the patterns for all material specimens in the train and test sets, except for one or two materials in the test set due to their under representation in the train set. For checking all the parity plots for the performance comparison of microstructure descriptors discussed in \figref{fig:mape_bar_strain_hardening_microstructure}, refer to Appendix Section \ref{app3}.

\subsection{Investigating texture descriptors }

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_strain_hardening_texture.pdf}
        \caption{}
        \label{fig:mape_bar_strain_hardening_texture}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_yield_stress_texture.pdf}
        \caption{}
        \label{fig:mape_bar_yield_stress_texture}
    \end{subfigure}
    \caption{Overall performance comparison of different texture descriptor based input features using mean absolute percentage error (MAPE) bar graph for predicting (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, using the input set $\mathbf{\Phi}$, at scale = $100 \,\upmu \text{m}$ using XGBoost regressor.}
    \label{fig:bar_texture}
\end{figure}


\begin{figure}[!h]
    \centering
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_embedding.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_embedding}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_embedding.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_embedding}
    \end{subfigure}
    \caption{True and predicted values of (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor using input feature $ ^\mathbf{VEC} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$ and input set $\mathbf{\Phi}$, at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_texture}
\end{figure}

An overall comparison of the performance of three different reduced texture descriptors in predicting $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ is shown in \figref{fig:bar_texture} using bar graph with MAPE values. For predicting $\mathbf{n}$ using the input set $\mathbf{\Phi}$, $^\mathbf{ISO} \bar{F}_l^{mn}$ performs the best with a test MAPE of 6.67\%, which is 2.41\% lower than the worst-performing $^\mathbf{ISO} g{\left( \varphi_1, \Phi, \varphi_2 \right)}$. Similarly, for predicting $\mathbf{\sigma_{y}}$ using the input set $\mathbf{\Phi}$, $^\mathbf{ISO} \bar{F}_l^{mn}$ works better with a test MAPE of 9.67\%, although the difference between this and the worst-performing $^\mathbf{ISO} g{\left( \varphi_1, \Phi, \varphi_2 \right)}$ is only 0.76\%. This indicates that isomap dimensionality reduction on the GSH coefficient vector works better than directly implementing it on the ODF for preserving the patterns in material texture that are indicative of $\mathbf{n}$ and $\mathbf{\sigma_{y}}$. This is due to the analytical dimensionality reduction inherent in the mathematical structure of GSH. Autoencoder-based vector embedding performed comparatively better than isomap for directly implementing dimensionality reduction on the ODF, with a test MAPE of 8.17\% for $\mathbf{n}$ and 10.28\% for $\mathbf{\sigma_{y}}$. However, $^\mathbf{ISO} \bar{F}l^{mn}$ is overall a better choice for a reduced texture descriptor in the prediction of both $\mathbf{n}$ and $\mathbf{\sigma_{y}}$. \figref{fig:parity_texture} shows the true vs. predicted values of $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ with the input set $\mathbf{\Phi}$ and vector embeddings of ODF, $^\mathbf{VEC} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$. There is a good agreement of the training data seen in \figref{fig:strain_hardening_XGB_embedding} for $\mathbf{n}$, but a comparatively worse fit to the training data in \figref{fig:yield_stress_XGB_embedding} for $\mathbf{\sigma_{y}}$. This can be attributed to the insufficiency in the dataset required to effectively learn the links between all the descriptors and $\mathbf{\sigma_{y}}$. However, overall, the XGBoost model is able to learn and effectively predict $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ with $^\mathbf{VEC} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$ as the texture descriptor, within a certain amount of accuracy, except for slight underfitting in \figref{fig:yield_stress_XGB_embedding}. Refer to Appendix Section \ref{app3} for all the parity plots comparing the performance for rest of the texture descriptors.

\subsection{Investigating length scales}
\label{sec8_4}

\begin{figure}[t]
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_strain_hardening_length.pdf}
        \caption{}
        \label{fig:mape_bar_strain_hardening_length}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/mape_bar_yield_stress_length.pdf}
        \caption{}
        \label{fig:mape_bar_yield_stress_length}
    \end{subfigure}
    \caption{Overall performance comparison at different length scales using mean absolute percentage error (MAPE) bar graph for predicting (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, using the input set $\mathbf{\Omega}$ using XGBoost regressor.}
    \label{fig:bar_regressor}
\end{figure}

\begin{figure}[!h]
    \centering
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_150.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_150}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_150.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_150}
    \end{subfigure}
    \caption{True and predicted values of (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor using input set $\mathbf{\Omega}$ at scale = $150 \,\upmu \text{m}$.}
    \label{fig:parity_regressor}
\end{figure}

\figref{fig:mape_bar_yield_stress_length} compares the performance of the combination of microstructure descriptors, $^\mathbf{PCA} f_3$ and $^\mathbf{ISO} \hat{G}$, calculated at length scales of $90$, $100$, $120$, and $150 \,\upmu \text{m}$ to predict $\mathbf{\sigma_{y}}$. Unlike previous investigations, this investigation shows that changing the length scale significantly impacts the test MAPE, with performance improving considerably at larger scales ($120$ and $150 \,\upmu \text{m}$). The test MAPE at $150 \,\upmu \text{m}$ is 7.01\%, which is 3.71\% better than at the scale of $90 \,\upmu \text{m}$. At larger length scales, relatively more grains and more grain boundary features are included in the images which used to calculate the reduced-order microstructure descriptors. The increased accuracy in $\mathbf{\sigma_{y}}$ prediction can be attributed to this additional information captured in the descriptors, which was missing at smaller length scales. This indicates that more grain boundary structure information is required to learn patterns for predicting $\mathbf{\sigma_{y}}$. For the prediction of $\mathbf{n}$, microstructure descriptors at a scale of $90 \,\upmu \text{m}$ perform the worst with a test MAPE of 8.59\%, while those at a scale of $150 \,\upmu \text{m}$ perform the best with a test MAPE of 5.35\% (see \figref{fig:mape_bar_strain_hardening_length}). Although the difference in test MAPE between scales $100$, $120$, and $150 \,\upmu \text{m}$ is only approximately 1.5\%, this indicates that sufficient information is captured between scales $100$ and $120 \,\upmu \text{m}$ for learning patterns that link to accurate $\mathbf{\hat{n}}$. This information is further enriched at $150 \,\upmu \text{m}$ with test MAPE of 5.35\%.

At a scale of $150 \,\upmu \text{m}$, there are fewer data points per material specimen due to the reduced number of images and tiles that can be cropped from those images. In \figref{fig:strain_hardening_XGB_150}, the best-performing hyperparameters are employed to train and test the prediction for $\mathbf{n}$. Although the model's test prediction is good, some underfitting is observed due to insufficient training data. Therefore, the robustness and stability of the XGBoost model in predicting $\mathbf{n}$ can be improved by increasing the number of material specimens and data points per specimen at a scale of $150 \,\upmu \text{m}$. A similar observation can be made for the prediction of $\mathbf{\sigma_{y}}$ at a scale of $150 \,\upmu \text{m}$. In this case, the training data fits relatively well along the best prediction line, but the same cannot be said for the test data due to insufficient representation of these test data points in the training data (see \figref{fig:yield_stress_XGB_150}). \ref{app3} contains the rest of parity plot comparisons at length scale $90$, $100$ and $120 \,\upmu \text{m}$. For predicting $\mathbf{\sigma_{y}}$, the issue of insufficient data is mitigated by the amount of grain boundary information contained in each data point.

\subsection{Feature importance using SHAP value analysis}
\label{sec8_5}
The SHAP value analysis as discussed in Section \ref{sec7_4} is visualized here using beeswarm plots of the best performing input and model configurations of XGBoost, to understand the importance of microstructure descriptors, statistics, and texture descriptors in predicting $\mathbf{n}$ and $\mathbf{\sigma_{y}}$. These SHAP beeswarm plots illustrate feature importance in an XGBoost regressor. Each dot represents a data point, with color indicating feature value (blue for low, red for high). Vertical stacking shows the concentration of data points at a SHAP value, while horizontal spread indicates the magnitude and direction of a feature’s impact on the model’s output. At a scale of $100 \,\upmu \text{m}$ (\figref{fig:SHAP_Strain_Plot_100}) and $150 \,\upmu \text{m}$ (\figref{fig:SHAP_Strain_Plot_150}), the descending order of importance for input features in predicting $\mathbf{n}$ remains consistent. This consistency suggests that sufficient information is captured even at the $100 ,\upmu \text{m}$ scale, with further enrichment at the $150 \,\upmu \text{m}$ scale, as evidenced by the lower test MAPE observed previously. In contrast, the SHAP value beeswarm plots for predicting $\mathbf{\sigma_{y}}$ (\figref{fig:SHAP_Yield_Plot_100} and \figref{fig:SHAP_Yield_Plot_150}) show distinct variations in the descending order of feature importance. Specifically, $^\mathbf{PCA} f_3$ at the $100 \,\upmu \text{m}$ scale does not capture significant information, whereas at the $150 \,\upmu \text{m}$ scale, it emerges as the third most important feature in predicting $\mathbf{\sigma_{y}}$.



\begin{figure}[t]
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/SHAP_Strain_Plot_100.pdf}
        \caption{}
        \label{fig:SHAP_Strain_Plot_100}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/SHAP_Yield_Plot_100.pdf}
        \caption{}
        \label{fig:SHAP_Yield_Plot_100}
    \end{subfigure}
    \caption{Feature dependence in the order of importance for (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor model based on SHAP value analysis, at scale = $100 \,\upmu \text{m}$.}
    \label{fig:shap_100}
\end{figure}

\begin{figure}[!h]
    \centering
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/SHAP_Strain_Plot_150.pdf}
        \caption{}
        \label{fig:SHAP_Strain_Plot_150}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.495\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/SHAP_Yield_Plot_150.pdf}
        \caption{}
        \label{fig:SHAP_Yield_Plot_150}
    \end{subfigure}
    \caption{Feature dependence in the order of importance for (a) Strain Hardening Exponent, $\mathbf{n}$ and (b) Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor model based on SHAP value analysis, at scale = $150 \,\upmu \text{m}$.}
    \label{fig:shap_150}
\end{figure}

The texture descriptor, $^\mathbf{ISO} \bar{F}_l^{mn}$, is the most important feature in predicting both $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ for this pipeline. This observation is well-supported by material science principles and literature on HCP materials, such as magnesium alloys. Magnesium alloys (e.g., ZM21, AZ31) exhibit strong basal texture due to high basal pole intensities aligned near the normal direction, which significantly influences yield stress. Small deviations from this principal alignment then result in higher yield stress in the transverse direction and lower yield stress in the rolling direction due to high activation ability of basal slip \citep{BOHLEN20072101}. Additionally, the strong basal texture in Mg-alloys and tensile twinning leads to slower strain hardening rates across different orientations \citep{Guo, Lou}. The dependence of uniaxial properties, $\mathbf{n}$ and $\mathbf{\sigma_{y}}$, on the normal and material direction makes the statistics of aspect ratio distribution the second most important feature in both predictions. The influence of grain boundary features in restricting twin growth and associated reorientation of the grain orientation, as well as their role in cross slip and deformation-carrying dislocations, underscores the importance of microstructure descriptors. Specifically, $^\mathbf{PCA} f_3$ is shown to be significant in predicting $\mathbf{n}$, as illustrated in \figref{fig:shap_150} \citep{ravaji}. Both twin boundaries and grain boundaries affect tensile yield stress, with the Hall-Petch relationship serving as a fundamental principle. Dislocation pile-up and the activation of multislip at grain boundaries influence yield stress \citep{YuHuihuiandXin, FU20012567}. This is why $^\mathbf{PCA} f_3$ emerges as the third most important feature in predicting $\mathbf{\sigma_{y}}$. However, in our dataset, the strong basal texture of Mg-alloys means that the texture descriptor predominantly drives both predictions. Another important observation from \figref{fig:shap_150} is that the microstructure descriptor, $^\mathbf{ISO} \hat{G}$, has a greater impact on predicting $\mathbf{n}$ and a lesser impact on predicting $\mathbf{\sigma_{y}}$. This can be attributed to the fact that the style and texture of grain boundaries influence slip more significantly, which in turn affects $\mathbf{n}$, while having a lesser impact on $\mathbf{\sigma_{y}}$.

\section{Conclusion}
\label{sec9}

In this study, a machine learning pipeline for a generalized approach to property prediction from microstructure and texture descriptors is developed, implemented, and successfully tested. A data extraction and preprocessing pipeline is established to create a coherent database of extruded Mg-alloys, as detailed in \tabref{tab:matdata}. An image processing and deep learning-based workflow for microstructure binarization and grain statistic calculation is applied to OM microstructure images, which proved to be very useful for making the data more uniform and coherent for further machine learning operations. The XRD pole plots from the database are used to calculate the ODF, which is efficiently represented using GSH coefficients. The images, ODF, and GSH representation of ODF are further employed to compute statistical descriptors for microstructure and texture. Dimensionality reduction techniques, such as PCA, Isomap, and autoencoder-based vector embedding, are then applied to these descriptors according to their suitability. Finally, their performance in predicting $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ is evaluated using GP, XGBoost, and MLP regressors. Several key findings are observed in this study, and they are as follows:

 \begin{itemize}
     \item The XGBoost model emerged as the best non-linear regressor according to test and train MAPE for the database. This performance can be attributed to the complexity of the input features and the sparse nature of the dataset. The GP model came close but suffered from underfitting.
     \item The combination of PCA-reduced three point spatial correlation $^\mathbf{PCA} f_3$ and Isomap-reduced gram matrices $^\mathbf{ISO} \hat{G}$ served as the best choice for microstructure descriptors. 
     \item The Isomap-reduced GSH coefficient vector, $^\mathbf{ISO} \bar{F}_l^{mn}$, served as the texture descriptor. Generalized Spherical Harmonics represented the ODF most efficiently for preserving information after dimensionality reduction.
     \item The texture vector and aspect ratio distribution ($\mathbf{AR}$) contributed more prominently to the prediction of $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ compared to the microstructure descriptor.
     \item Larger length scales ($120$ and $150 \,\upmu \text{m}$) are more suitable for effectively calculating microstructure statistics for property prediction (primarily for $\mathbf{\sigma_{y}}$)  because they capture more information.
     \item Finally, this structure-property pipeline achieved a high predictive performance for unseen material specimens of Mg-alloys, with a MAPE of 6.67\% for predicting the strain hardening exponent ($\mathbf{n}$) and 7.01\% for predicting the tensile yield stress ($\mathbf{\sigma_{y}}$).
     
 \end{itemize}

The performance metrics achieved through this pipeline surpass the current state-of-the-art benchmarks for studies based on training and testing experimental data, rather than data generated via simulation. This paper not only presents a methodology for predicting the properties of extruded Mg-alloys but also provides a template for optimizing this pipeline. This approach helps generalize the methodology for other material types, revealing correlations between properties and microstructure or texture descriptors. Certain limitations were also present in this work, such as the insufficient representation of certain Mg-alloys in the training data. Additionally, including process parameters and alloy composition in the descriptors could enhance the robustness and accuracy of predicting mechanical properties. Increased efforts in collecting data specifically for the purpose to serve such pipeline would increase the quality and quantity of the training data. Another limitation of this study is the absence of any physics-based aspects in property prediction, as the study is entirely data-driven. Another future scope of this study could involve enhancing the dataset with physics-based simulations to create a hybrid data-driven and physics-based property prediction pipeline.

\section*{Acknowledgements}
This work was supported by Helmholtz Center Hereon, Leuphana Universität Lüneburg, and Hamburg University of Technology (TUHH). The authors would like to thank Dr. Maria Nienaber for her invaluable guidance in magnesium alloys and mechanical property analysis, and Dr. Nowfal Al-Hamdany for his assistance in texture analysis of Magnesium alloys.

%% The Appendices part is started with the command \appendix;
%% appendix sections are then done as normal sections
\appendix

\section{Prediction perfor+mance for $\mathbf{\sigma_{max}}$, $\mathbf{\varepsilon_{pu}}$ and $\mathbf{\varepsilon_{pf}}$}
\label{app1}

In this section, the prediction performance for ultimate tensile stress ($\mathbf{\sigma_{max}}$), uniform strain ($\mathbf{\varepsilon_{pu}}$), and fracture strain ($\mathbf{\varepsilon_{pf}}$) is presented for the best-performing regressor, XGBoost. The prediction performance is demonstrated using the best-performing microstructure and texture descriptors in the input feature set $\mathbf{\Omega}$ at a scale of $120 ,\upmu \text{m}$. Hyperparameter selection was conducted as discussed earlier in the paper, and the configuration with the most optimal MAPE and MAE is shown in \figref{fig:parity_other}.

The prediction of $\mathbf{\sigma_{max}}$ shows a good test MAPE value of 4.24\%, but the $R^2$ value suffers due to some material specimens in the test set not being adequately learned during training. A similar issue is observed for the prediction of $\mathbf{\varepsilon_{pu}}$, as the data available at a scale of $120 ,\upmu \text{m}$ is insufficient to capture the values of each material specimen in the dataset. While the test MAPE and $R^2$ values for $\mathbf{\varepsilon_{pf}}$ are relatively good, further enhancement of the dataset would still be beneficial.


\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/ultimate_stress_XGB_120.pdf}
        \caption{}
        \label{fig:ultimate_stress_XGB_120}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/uniform_strain_XGB_120.pdf}
        \caption{}
        \label{fig:uniform_strain_XGB_120}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.475\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/fracture_strain_XGB_120.pdf}
        \caption{}
        \label{fig:fracture_strain_XGB_120}
    \end{subfigure}
    \caption{True and predicted values of (a) Ultimate tensile stress, $\mathbf{\sigma_{max}}$, (b) Uniform strain, $\mathbf{\varepsilon_{pu}}$, and (c) Fracture strain $\mathbf{\varepsilon_{pf}}$, for XGBoost regressor using input set $\mathbf{\Omega}$ at scale = $120 \,\upmu \text{m}$.}
    \label{fig:parity_other}
\end{figure}




\section{Performance measures for all investigations}
\label{app2}

In this section all the performance measures are summarized for XGBoost, GP and MLP models in terms of train and test MAE and MAPE for prediction of $\mathbf{n}$ and $\mathbf{\sigma_{y}}$.

\subsection{XGBoost regressor}

\begin{table}[H]
\centering
\caption{\newline Input feature vectors and input sets along with their corresponding mean average percentage error (MAPE) and mean average error (MAE) of Strain Hardening Exponent, $\mathbf{n}$ and Yield Stress, $\mathbf{\sigma_{y}}$ for XGBoost regressor.}
\label{tab:autoencoder}
\resizebox{\textwidth}{!}{
\begin{tblr}{
  cell{1}{2} = {c=5}{},
  cell{1}{7} = {c=4}{},
  cell{2}{1} = {r=2}{},
  cell{2}{2} = {c=2}{},
  cell{2}{4} = {c=2}{},
  cell{2}{7} = {c=2}{},
  cell{2}{9} = {c=2}{},
  hline{1-2,4,13} = {-}{},
  hline{3} = {2-5,7-10}{},
}
              & Strain hardening exp. ($\mathbf{n}$) &      &       &      &  & Yield stress ($\mathbf{\sigma_{y}}$) &      &       &      \\
Configuration & MAPE    &      & MAE   &      &  & MAPE  &      & MAE   &      \\
              & Train   & Test & Train & Test &  & Train & Test & Train & Test \\
$( { }^{\mathbf{PCA}} f_2, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                               &  1.54\% &  7.92\% & 0.003 & 0.016 &  & 2.62\% & 9.36\%  & 3.52 MPa & 13.26 MPa \\
$( { }^{\mathbf{PCA}} f_3, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                               &  1.59\% &  9.50\% & 0.003 & 0.020 &  & 2.71\% & 9.26\%  & 3.69 MPa & 13.08 MPa \\
$( { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                           &  1.47\% &  6.97\% & 0.003 & 0.014 &  & 2.62\% & 9.51\%  & 3.59 MPa & 13.44 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$   &  1.73\% &  6.67\% & 0.004 & 0.013 &  & 2.69\% & 9.67\%  & 3.68 MPa & 13.65 MPa \\
$ ({ }^{\mathbf{VEC}} g_{( \varphi_1, \Phi, \varphi_2)} , \mathbf{\Phi})$ @ $100 \,\upmu \text{m}$&  1.44\% &  8.17\% & 0.003 & 0.018 &  & 2.69\% & 10.28\% & 3.54 MPa & 14.09 MPa \\
$ ({ }^{\mathbf{ISO}} g_{( \varphi_1, \Phi, \varphi_2)}, \mathbf{\Phi})$ @ $100 \,\upmu \text{m}$ &  2.05\% &  9.08\% & 0.004 & 0.020 &  & 2.92\% & 10.42\% & 3.89 MPa & 15.24 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $90 \,\upmu \text{m}$    &  1.29\% &  8.59\% & 0.003 & 0.018 &  & 2.26\% & 10.73\% & 3.09 MPa & 15.10 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $120 \,\upmu \text{m}$   &  2.13\% &  6.65\% & 0.004 & 0.014 &  & 2.04\% & 7.38\%  & 2.79 MPa & 10.92 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $150 \,\upmu \text{m}$   &  2.25\% &  5.35\% & 0.004 & 0.011 &  & 2.31\% & 7.01\%  & 3.05 MPa & 9.86 MPa      
\end{tblr}
}
\end{table}




\subsection{GP regressor}


\begin{table}[H]
\centering
\caption{\newline Input feature vectors and input sets along with their corresponding mean average percentage error (MAPE) and mean average error (MAE) of Strain Hardening Exponent, $\mathbf{n}$ and Yield Stress, $\mathbf{\sigma_{y}}$ for GP regressor.}
\label{tab:autoencoder}
\resizebox{\textwidth}{!}{
\begin{tblr}{
  cell{1}{2} = {c=5}{},
  cell{1}{7} = {c=4}{},
  cell{2}{1} = {r=2}{},
  cell{2}{2} = {c=2}{},
  cell{2}{4} = {c=2}{},
  cell{2}{7} = {c=2}{},
  cell{2}{9} = {c=2}{},
  hline{1-2,4,13} = {-}{},
  hline{3} = {2-5,7-10}{},
}
              & Strain hardening exp. ($\mathbf{n}$) &      &       &      &  & Yield stress ($\mathbf{\sigma_{y}}$) &      &       &      \\
Configuration & MAPE    &      & MAE   &      &  & MAPE  &      & MAE   &      \\
              & Train   & Test & Train & Test &  & Train & Test & Train & Test \\
$( { }^{\mathbf{PCA}} f_2, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                               &  3.61\% &  9.34\% & 0.008 & 0.024  &  & 3.87\% & 12.74\%  & 5.39 MPa & 12.31 MPa \\
$( { }^{\mathbf{PCA}} f_3, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                               &  3.28\% &  9.42\% & 0.007 & 0.024  &  & 3.51\% & 12.44\%  & 4.89 MPa & 11.99 MPa \\
$( { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                           &  4.12\% &  9.92\% & 0.009 & 0.025  &  & 4.43\% & 13.06\%  & 6.14 MPa & 12.77 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$   &  3.09\% &  10.60\% & 0.007 & 0.026 &  & 3.26\% & 13.15\%  & 4.61 MPa & 12.97 MPa \\
$ ({ }^{\mathbf{VEC}} g_{( \varphi_1, \Phi, \varphi_2)} , \mathbf{\Phi})$ @ $100 \,\upmu \text{m}$&  3.68\% &  8.68\% & 0.008 & 0.021  &  & 4.19\% & 11.96\%  & 6.07 MPa & 12.39 MPa \\
$ ({ }^{\mathbf{ISO}} g_{( \varphi_1, \Phi, \varphi_2)}, \mathbf{\Phi})$ @ $100 \,\upmu \text{m}$ &  1.85\% &  10.13\% & 0.004 & 0.025 &  & 2.28\% & 16.14\%  & 3.58 MPa & 16.28 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $90 \,\upmu \text{m}$    &  3.03\% &  9.78\% & 0.007 & 0.023  &  & 3.35\% & 12.35\%  & 4.91 MPa & 12.51 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $120 \,\upmu \text{m}$   &  2.99\% &  10.18\% & 0.007 & 0.026 &  & 3.05\% & 14.25\%  & 4.22 MPa & 12.91 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $150 \,\upmu \text{m}$   &  2.81\% &  9.80\% & 0.006 & 0.026  &  & 2.70\% & 14.71\%  & 3.68 MPa & 12.21 MPa     
\end{tblr}
}
\end{table}



\subsection{MLP regressors}


\begin{table}[H]
\centering
\caption{\newline Input feature vectors and input sets along with their corresponding mean average percentage error (MAPE) and mean average error (MAE) of Strain Hardening Exponent, $\mathbf{n}$ and Yield Stress, $\mathbf{\sigma_{y}}$ for MLP regressor.}
\label{tab:autoencoder}
\resizebox{\textwidth}{!}{
\begin{tblr}{
  cell{1}{2} = {c=5}{},
  cell{1}{7} = {c=4}{},
  cell{2}{1} = {r=2}{},
  cell{2}{2} = {c=2}{},
  cell{2}{4} = {c=2}{},
  cell{2}{7} = {c=2}{},
  cell{2}{9} = {c=2}{},
  hline{1-2,4,13} = {-}{},
  hline{3} = {2-5,7-10}{},
}
              & Strain hardening exp. ($\mathbf{n}$) &      &       &      &  & Yield stress ($\mathbf{\sigma_{y}}$) &      &       &      \\
Configuration & MAPE    &      & MAE   &      &  & MAPE  &      & MAE   &      \\
              & Train   & Test & Train & Test &  & Train & Test & Train & Test \\
$( { }^{\mathbf{PCA}} f_2, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                                &  6.63\% &  8.19\% & 0.015 & 0.017   &  & 3.49\% & 14.82\%  & 5.04 MPa & 20.91 MPa \\
$( { }^{\mathbf{PCA}} f_3, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                                &  2.95\% &  11.53\% & 0.006 & 0.023  &  & 4.36\% & 9.81\%  & 6.09 MPa & 13.50 MPa \\
$( { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$                            &  2.66\% &  8.49\% & 0.006 & 0.017   &  & 4.08\% & 14.88\%  & 5.94 MPa & 12.57 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $100 \,\upmu \text{m}$    &  4.47\% &  8.60\% & 0.010 & 0.018   &  & 4.36\% & 9.98\%  & 6.18 MPa & 13.78 MPa \\
$ ({ }^{\mathbf{VEC}} g_{( \varphi_1, \Phi, \varphi_2)} , \mathbf{\Phi})$ @ $100 \,\upmu \text{m}$ &  4.85\% &  8.22\% & 0.010 & 0.018   &  & 6.69\% & 12.91\%  & 9.40 MPa & 17.01 MPa \\
$ ({ }^{\mathbf{ISO}} g_{( \varphi_1, \Phi, \varphi_2)}, \mathbf{\Phi})$ @ $100 \,\upmu \text{m}$  &  4.68\% &  8.82\% & 0.010 & 0.017   &  & 4.04\% & 13.18\%  & 6.12 MPa & 18.11 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $90 \,\upmu \text{m}$     &  4.33\% &  10.74\% & 0.009 & 0.022  &  & 3.00\% & 10.78\%  & 4.73 MPa & 15.26 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $120 \,\upmu \text{m}$    &  4.09\% &  10.59\% & 0.009 & 0.020  &  & 5.01\% & 12.40\%  & 6.74 MPa & 16.78 MPa \\
$( { }^{\mathbf{PCA}} f_3, { }^{\mathbf{ISO}} \hat{G}, \mathbf{\Psi})$ @ $150 \,\upmu \text{m}$    &  3.76\% &  10.39\% & 0.008 & 0.021  &  & 5.10\% & 10.75\%  & 6.74 MPa & 14.26 MPa
\end{tblr}
}
\end{table}



\section{Remaining parity plots for the investigations}
\label{app3}

In this section, we include the comparison of true and predicted values of $\mathbf{n}$ and $\mathbf{\sigma_{y}}$ for the investigations conducted in this paper for microstructure, texture and length scales. These comparisons were not shown in the main paper for simplicity.

\subsection{Microstructure descriptors}

Strain Hardening Exponent, $\mathbf{n}$

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_2point_GSH.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_2point_GSH}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_3point_GSH.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_3point_GSH}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_gram_GSH.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_gram_GSH}
    \end{subfigure}
    \caption{True and predicted values of Strain Hardening Exponent, $\mathbf{n}$, for XGBoost regressor using input features (a) $ ^\mathbf{PCA} f_2$, (b)  $ ^\mathbf{PCA} f_3$, and (c) $ ^\mathbf{ISO} \hat{G}$ with input set $\mathbf{\Psi}$ at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_microstructur_appendix_strain}
\end{figure}

Yield Stress, $\mathbf{\sigma_{y}}$

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_3point_GSH.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_3point_GSH}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_gram_GSH.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_gram_GSH}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_3point_gram_GSH.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_3point_gram_GSH}
    \end{subfigure}
    \caption{True and predicted values of Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor using input features (a) $ ^\mathbf{PCA} f_3$, (b)  $ ^\mathbf{ISO} \hat{G}$, and (c) $ ^\mathbf{PCA} f_3 $, $ ^\mathbf{ISO} \hat{G}$ with input set $\mathbf{\Psi}$ at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_microstructur_appendix_yield}
\end{figure}


\subsection{Texture descriptors}

Strain Hardening Exponent, $\mathbf{n}$

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_GSH.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_GSH}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_isomap.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_isomap}
    \end{subfigure}
    \caption{True and predicted values of Strain Hardening Exponent, $\mathbf{n}$, for XGBoost regressor using input features (a) $ ^\mathbf{ISO} \bar{F}_l^{mn}$, and (b) $ ^\mathbf{ISO} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$ with input set $\mathbf{\Phi}$ at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_texture_appendix_strain}
\end{figure}

Yield Stress, $\mathbf{\sigma_{y}}$

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_GSH.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_GSH}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_isomap.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_isomap}
    \end{subfigure}
    \caption{True and predicted values of Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor using input features (a) $ ^\mathbf{ISO} \bar{F}_l^{mn}$, and (b) $ ^\mathbf{ISO} g_{\left( \varphi_1, \Phi, \varphi_2 \right)}$ with input set $\mathbf{\Phi}$ at scale = $100 \,\upmu \text{m}$.}
    \label{fig:parity_texture_appendix_yield}
\end{figure}


\subsection{Length scale}

Strain Hardening Exponent, $\mathbf{n}$

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_90.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_90}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_100.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_100}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/strain_hardening_XGB_120.pdf}
        \caption{}
        \label{fig:strain_hardening_XGB_120}
    \end{subfigure}
    \caption{True and predicted values of Strain Hardening Exponent, $\mathbf{n}$, for XGBoost regressor using with input set $\mathbf{\Psi}$ at scale (a) $90$, (b) $100$, and (c) $120 \,\upmu \text{m}$.}
    \label{fig:parity_length_appendix_strain}
\end{figure}

Yield Stress, $\mathbf{\sigma_{y}}$

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_90.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_90}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_100.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_100}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.45\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Figures/results/yield_stress_XGB_120.pdf}
        \caption{}
        \label{fig:yield_stress_XGB_120}
    \end{subfigure}
    \caption{True and predicted values of Yield Stress, $\mathbf{\sigma_{y}}$, for XGBoost regressor using with input set $\mathbf{\Psi}$ at scale (a) $90$, (b) $100$, and (c) $120 \,\upmu \text{m}$.}
    \label{fig:parity_length_appendix_yield}
\end{figure}


  \bibliographystyle{unsrt} 
  \bibliography{no_synthetic_data}


%% For citations use: 
%%       \citet{<label>} ==> Lamport (1994)
%%       \citep{<label>} ==> (Lamport, 1994)

%% If you have bib database file and want bibtex to generate the
%% bibitems, please use
%%
%%  \bibliographystyle{elsarticle-harv} 
%%  \bibliography{<your bibdatabase>}

%% else use the following coding to input the bibitems directly in the
%% TeX file.

%% Refer following link for more details about bibliography and citations.
%% https://en.wikibooks.org/wiki/LaTeX/Bibliography_Management

% \begin{thebibliography}{00}

% %% For authoryear reference style
% %% \bibitem[Author(year)]{label}
% %% Text of bibliographic item

% \bibitem[Lamport(1994)]{lamport94}
%   Leslie Lamport,
%   \textit{\LaTeX: a document preparation system},
%   Addison Wesley, Massachusetts,
%   2nd edition,
%   1994.

% \end{thebibliography}
\end{document}

\endinput
%%
%% End of file `elsarticle-template-harv.tex'.


