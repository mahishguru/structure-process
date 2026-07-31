\documentclass{article}

% NeurIPS 2026 style --- anonymized submission (default = main, double-blind).
% Switch to [preprint] for arXiv or [final] for camera-ready.
% Numeric citation style: \citep -> [3], \citep{a,b,c} -> [3, 5, 7] (compressed/sorted).
\PassOptionsToPackage{numbers,sort&compress}{natbib}
\usepackage{neurips_2026}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{enumitem}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{nicefrac}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{wrapfig}
\usepackage{float}
\usepackage[table]{xcolor}

% Caption package is needed for \captionof in combined figure/table blocks.
\usepackage{caption}
% NeurIPS-risky spacing tweaks kept disabled for template compliance.
\captionsetup[figure]{aboveskip=0.3em}
\captionsetup[table]{aboveskip=0pt,belowskip=1pt}
% \setlength{\abovecaptionskip}{0pt}
% \setlength{\belowcaptionskip}{0pt}
\setlength{\textfloatsep}{6pt plus 1pt minus 0pt}
% \setlength{\floatsep}{6pt plus 1pt minus 1pt}
% \setlength{\intextsep}{6pt plus 1pt minus 1pt}
% \raggedbottom

% --- Convenience macros (used throughout) ---
\newcommand{\Co}{\textsc{Co-PiLOT}}
\newcommand{\meridian}{\textsc{Meridian}}
\newcommand{\dante}{\textsc{Dante}}
\newcommand{\turbo}{\textsc{TuRBO}}
\newcommand{\baxus}{\textsc{BAxUS}}
\newcommand{\saasbo}{\textsc{SaasBO}}
\newcommand{\damask}{\textsc{Damask}}
\newcommand{\dreamthreed}{\textsc{Dream.3D}}
\newcommand{\dkl}{DKL-GP}
\newcommand{\Z}{\mathcal{Z}}
\newcommand{\X}{\mathcal{X}}
\newcommand{\Dset}{\mathcal{D}_t}
\newcommand{\Enc}{\mathcal{E}}
\newcommand{\Dec}{\mathcal{D}}
\newcommand{\Oracle}{\mathcal{S}}
\newcommand{\Opt}{\mathcal{O}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\norm}[1]{\left\lVert#1\right\rVert}

% Common shortened symbols.
\newcommand{\sigy}{\sigma_y}
\newcommand{\sigu}{\sigma_u}

\title{\Co{}: Constrained Physics-Informed Latent Optimization for Target-Driven Inverse Design}

\author{%
  Anonymous Authors \\
  Anonymous Affiliation \\
  \texttt{anonymous@anonymous.org}
}

\begin{document}

\maketitle

%==============================================================================
\begin{abstract}
Inverse design of physical systems (molecules, devices, microstructures) often reduces to optimizing a high-dimensional structure against an expensive black-box simulator. Direct search is difficult because the space is non-Euclidean, feasibility is hard to encode, and each evaluation is expensive. We present \Co{}, a latent optimization approach that maps candidates through a generative encoder--decoder, uses the decoder as a learned validity prior, and searches the latent space with physics-informed black-box optimization. The framework is applied on the inverse design of magnesium alloy microstructure/texture. We develop a vision transformer based--encoder; paired with latent diffusion, diffusion transformer and rectified-flow transformer--based decoders on $\sim80{,}000$ EBSD-derived microstructure dataset to learn a minimal bottleneck, $z$. The ViT-FMDiT model ($z$=$768$) reconstructs high-fidelity microstructure images (FID $23.19$, MS-SSIM $0.178$), which our self-segmenting orientation codec converts into input grids for crystal plasticity solver. Finally, we introduce \meridian{}, an active latent optimizer driven by deep-kernel Gaussian-process uncertainty, feasibility prediction, active trust regions, and target-aware acquisition. Within the evaluation budget, the ViT-FMDiT and \meridian{} combination yields the best target-driven objective score, outperforming \dante{}, \turbo{}, and \baxus{} by $\sim\!6\%$ in relative error on the same decoder.
\end{abstract}

%==============================================================================
\section{Introduction}
\label{sec:intro}

Across science and engineering, the same template recurs: \emph{a designer is given a target behaviour and must find a physical object that realises it}. A medicinal chemist is given a binding-affinity profile and must find a molecule whose density-functional-theory (DFT) energetics match it~\citep{gomez2018, Yoo2023,Axelrod2022}; a photonics engineer is given a target far-field radiation pattern and must find a metasurface whose Maxwell-solver response matches it~\citep{Molesky2018,Zhou2021}; a crystallographer is given a desired band gap and must find a periodic crystal whose first-principles spectrum matches it~\citep{xie2022,zeni2025,jiao2023}. Inverse design asks: given a target property vector $p^\star$, find a design $x^\star \in \X$ such that $f(x^\star) \approx p^\star$, where $f$ is an expensive black-box oracle (e.g., a physical assay or simulator) and a feasibility constraint $g(x) \le 0$ encodes stability requirements. Three structural obstacles make direct search on $\X$ intractable. \emph{(i)}~$\X$ is discreet and non-Euclidean (graphs of atoms, fields of crystallographic orientations), so gradient methods do not apply through a simulator that has no usable adjoint~\citep{Liu2023}. \emph{(ii)}~most points in $\X$ are physically invalid, and feasibility cannot be expressed as a simple box, so unguided sampling almost never yields a candidate that oracle will run to convergence~\citep{Lu_2022}. \emph{(iii)}~Each evaluation $f(x)$ costs hours, with $10^2$--$10^3$ evaluation budgets that rules out brute-force sampling, GAN-inversion-by-backpropagation~\citep{creswell2018,li2026towards}, and high-dimensional Bayesian optimization (BO) over the raw space~\citep{eriksson2019,papenmeier2022, Frazier_2015}. Prior work side-steps the simulator by training a fast surrogate~\citep{yang2018,cang2018,xie2022,zeni2025}, trading simulator cost for surrogate-fidelity risk that breaks whenever the surrogate is asked to extrapolate past its training distribution.

To ground this framework, we tackle an industrial and scientific challenge that falls entirely out of the 'cheap-oracle' assumption dominating existing latent-optimization research: \emph{target-driven inverse design of polycrystalline Magnesium (Mg) alloy microstructures and textures}. The protagonist is the engineer with a target stress--strain curve, eg. a structural designer specifying a Mg-alloy AZ31 component for an automotive crash member~\citep{Abbott2016}; a clinician specifying a bioresorbable Mg--Gd bone screw whose yield strength must match cortical bone over a degradation window~\citep{Chiu}; an aerospace designer chasing an anisotropic high-toughness texture in a structural panel~\citep{Bai2023}. In every case the deliverable is not a microstructure \emph{image} but an arrangement of grains and crystallographic orientations that, when simulated through a crystal-plasticity (CP) oracle such as \damask{}~\citep{roters2019}, returns a property tuple $(\sigy, n, K, \sigu)$ within tolerance of the target. Microstructure inverse design is a hard problem because the design space comprises of grains with symmetric-crystallographic orientaions~\citep{Mason2009,Tuomo} and anisotropic, twin-dominated, and texture-coupled plasticity which causes visually similar microstructures to yield vastly different stress--strain responses~\citep{guru2025}; and a single $300{\times}300$ CP simulation takes $\sim\!6$ minutes and risks divergence~\citep{roters2019}.

\textbf{Why no off-the-shelf recipe applies?} To bypass the intractability of raw search space, one could theoretically map the problem into a continuous latent search space with a pretrained generator $\Dec : \R^d \to \X$. In such a setup, decoded samples lie near a learned manifold, the latent prior yields a free box constraint $\Z = [-B, B]^d$, and the optimization objective $J \circ f \circ \Dec$ is well-defined. This theoretically enables pure target-driven design, where optimal orientations and grain morphologies emerge directly from stress--strain requirements~\citep{Xiong2016}. \emph{In practice, however,} existing studies assume cheap oracles~\citep{tripp2020,gomez2018,maus2022,stanton2022}, high-dimensional BO lacks generative priors~\citep{eriksson2019,papenmeier2022,eriksson2021}, and recent crystal models~\citep{xie2022,zeni2025,jiao2023} are built for sampling, not inverse optimization. None can reliably navigate an expensive, occasionally divergent simulator while enforcing valid material-grain orientation fields.

\textbf{Contributions}. We present \Co{} (\textbf{Co}nstrained \textbf{P}hysics-\textbf{i}nformed \textbf{L}atent \textbf{O}ptimization for \textbf{T}arget-driven inverse design), a framework that lifts inverse design into a pretrained generative latent space and runs a constrained physics-informed black-box optimizer there.
\begin{enumerate}[leftmargin=1.2em,itemsep=-2pt,topsep=2pt]
  \item \textbf{General framework} (Sec.~\ref{sec:method}). We formulate inverse design as a 5-tuple $(\Dec, f, J, \mathcal{C}, \Opt)$ with independently swappable decoder, oracle, objective, constraint set, and optimizer. The framework is domain-agnostic; exemplified by a materials design instantiation.
    \item \textbf{Microstructure encoder--decoder family} (Sec.~\ref{sec:enc-dec}). A shared Vision Transformer (ViT) based encoder is paired with a primary Flow Matching Diffusion Transfomer (FMDiT) based decoder, trained jointly on $81{,}756$ EBSD-derived microstructures across 17 Mg-alloy classes.
    \item \textbf{Orientation codec} (Sec.~\ref{sec:codec}). An invertible map between RGB images and orientation fields with a closed-form decode and a grain-segmentation \dreamthreed{} writer that is used by the CP solver. It achieves ${\sim}0.7^\circ$ round-trip misorientation, below the $5^\circ$ grain-boundary threshold~\citep{readshockley1950, groeber2014}.
    \item \textbf{\meridian{}, a latent optimizer} (Sec.~\ref{sec:meridian}). We developed a combination of a deep-kernel Guassian Process surrogate, a shared-trunk feasibility classifier, an active-subspace trust region, and a DPP batch selector; benchmarked against \dante{}, \turbo{}, and \baxus{}, where ViT-FMDiT-$768$ with \meridian{} reaches the best target-driven and toughness-weighted objectives.
\end{enumerate}

Although demonstrated on polycrystalline AZ31 microstructure design, the 5-tuple abstraction is agnostic to the physical domain: any setting with a pretrained decoder for $\X$ and a slow oracle scoring $\Dec(z)$ is in scope, eg. molecular optimization with a VAE~\citep{gomez2018} and a DFT oracle, photonic-device design with a pretrained device generator and an EM solver~\citep{Yeung2023}, or topology optimization with a pretrained GAN and an FEM oracle~\citep{nie2020}. The pretrained encoder, the three decoders, the 17-class orientation codec and the \meridian{} optimizer calibrated to the noise level of the headless \damask{} harness will be release alongside this contribution for the latent-optimization community.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/Neurips_fig1_final.png}
   \caption{\textbf{\Co{}.} (a)~General framework and its materials instantiation (b): a Mg-alloy latent $z^\star \in \Z$ is passed through a decoder $\Dec$ like \textsc{DiT} or \textsc{FM-DiT}, bridged via an orientation codec ($\Psi$) to a \texttt{.dream3d} grid, scored by a crystal plasticity oracle $f$, and reduced to a fitness $J$ ($J_{V1}/J_{V2}$) that the \meridian{} optimizer $\Opt$ feeds back as $z_\mathbf{new}$ for the next iteration.}
  \label{fig:hero}
\end{figure}

%==============================================================================
\section{Related work}
\label{sec:related}

\textbf{High-dimensional latent optimization.} Latent-space optimization was popularized by \citet{gomez2018}, who showed that a learned continuous representation can make structured design amenable to gradient-based and Bayesian optimization. Follow-up latent-BO methods add weighted retraining, trust regions, and constrained or multi-objective variants~\citep{tripp2020,maus2022,stanton2022,zeng2024antibody}, but they are largely evaluated with cheap neural property predictors rather than expensive physical simulators. High-dimensional BO methods such as \turbo{}, \baxus{}, and \saasbo{} improve sample efficiency in large ambient spaces,\citep{eriksson2019,papenmeier2022,eriksson2021} yet they do not exploit generative priors or explicitly address simulator divergence and structured feasibility, which are central in our setting. Classical black-box optimizers such as CMA-ES remain strong generic baselines~\citep{hansen2001}, but full covariance adaptation scales poorly in high dimensions and is difficult to justify under our $\sim\!100$--$200$ evaluation budget.

\textbf{Generative inverse design of materials.} Generative models have become a standard tool for inverse design in materials and related physical systems~\citep{Lu_2022}. Early materials work used GANs to generate microstructures and couple them to regressors or Bayesian optimization~\citep{yang2018,cang2018}, while more recent studies perform inverse design of dual-phase steel microstructures with generative models and BO~\citep{KusampudiDiehl2023}, inverse design of spinodoid architected materials in the small-data regime via BO~\citep{Rassloff2026}, or use diffusion models for nonlinear mechanical metamaterials and multi-material structures~\citep{ParkKushwaha2024,Zheng2026}. In crystals, CDVAE, DiffCSP, and MatterGen show that modern generative models can capture complex structure distributions~\citep{xie2022,jiao2023,zeni2025}, but these methods are primarily built for sampling or one-shot conditional generation rather than iterative optimization under a strict expensive-oracle budget.

\textbf{Black-box optimization around generators.} Our work is closest in spirit to methods that place an outer optimization loop around a pretrained generator, as in latent BO for molecules~\citep{gomez2018,tripp2020}, photonic inverse design with neural generators and EM solvers~\citep{Molesky2018,Zhou2021,Yeung2023}, and GAN-based topology optimization with FEM~\citep{nie2020,ParkKushwaha2024}. The key gap is that these settings typically rely on cheaper solvers, smoother objectives, or direct conditional generation. \Co{} instead targets the harsher regime of constrained latent optimization with a slow, occasionally divergent physics solver, and therefore emphasizes calibrated surrogates, feasibility modeling, and robust search in pretrained latent spaces.

%==============================================================================
\section{Method}
\label{sec:method}

The central act of \Co{} is the tight coupling of a pretrained microstructure decoder $\Dec$ (Sec.~\ref{sec:enc-dec}) with a physics-informed optimizer $\Opt$ (Sec.~\ref{sec:pipeline}) via an orientation codec ensuring crystallographic validity (Sec.~\ref{sec:codec}). Operating hand-in-hand inside a single loop (Fig.~\ref{fig:hero}), the decoder turns a latent vector $z{\in}\Z$ into a candidate design, while the optimizer proposes the next $z$ to query based on a scalar fitness $J$ returned by an oracle simulator $f$ (Sec.~\ref{sec:optimizers}).

\textbf{Problem formulation}. Let $\X$ denote a structured design space (microstructures, molecules, \dots), $f : \X \to \R^k$ an expensive oracle returning a property tuple $p = f(x) \in \R^k$, $p^\star \in \R^k$ a target vector, $J : \R^k \to \R$ a scalar objective combining target distance and physics priors, and $g : \X \to \R^m$ feasibility constraints. Given a pretrained encoder--decoder pair $(\Enc, \Dec)$ with $\Dec : \R^d \to \X$, \Co{} maps $\X$ into the latent box $\Z = [-B, B]^d$ inherited from the latent prior and solves
\begin{equation}
  z^\star \;=\; \arg\max_{z \in \Z}\; J\bigl(f(\Dec(z));\, p^\star\bigr)
                  \;-\; \lambda \cdot \mathrm{ReLU}\!\bigl(g(\Dec(z))\bigr),
  \qquad x^\star = \Dec(z^\star).
  \label{eq:latent-inverse-design}
\end{equation}
This reformulation has three structural advantages.
(i)~Deliberate validity as $\Dec(z)$ is always a plausible design.
(ii)~Box constraint gives a natural prior $\Z$.
(iii)~Regularised search space as the decoder restricts the optimizer to a learned manifold, resulting in structured modeling target than direct search over the raw $\X$. The materials instantiation (Fig.~\ref{fig:hero}b) dictates the anatomy of this section

\subsection{Implementation of Encoder--Decoder ($\Dec$) }
\label{sec:enc-dec}

\Co{} is built around a continually \emph{pre-trained} stack: a ViT-H/14 encoder~\citep{radford2021clip,schuhmann2022laion5b} and diffusion decoders- SDXL~\citep{podell2023sdxl}, SD3.5~\citep{esser2024sd3}, and DiT-XL/2~\citep{peebles2023dit}, that are pre-trained on massive natural-image datasets. We posit this prior is very effective for microstructure reconstruction on a very thin bottlneck $z$. Even a $d{=}512$ bottleneck (${\sim}1500{\times}$ compression) recovers grain morphology, crystallographic colour, and property statistics that are compareable to a from-scratch decoder (Sec.~\ref{sec:e1}).


\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/Neurips_Figure2.png}
  \caption{(a)~\textbf{Encoder $\Enc$}, ViT-H/14 trunk $\to$ $\textsc{Attention Pooler}$ with $z\!\in\!\R^{d}$. (b)~\textbf{Decoder $\Dec$}, Frozen SD3.5 MMDiT conditioned via a $\textsc{LatentToTokens}$ adaptor and $\textsc{Pooled Projection}$.}
  \label{fig:archs}
\end{figure}

\textbf{A single shared encoder architecture.} A ViT-H/14 trunk maps $512^2{\times}3$ orientation-as-RGB microstructure images to $1369$ patch tokens plus \texttt{[CLS]} token. 2B weights are reused with the lower $16$ blocks frozen and the upper $16$ unfrozen (Fig.~\ref{fig:archs}(a)), retaining the natural-image prior while letting the top of the trunk specialise to EBSD imagery. The \texttt{[CLS]}-only readout is replaced by a four-query \textsc{Attention Pooler} (cross-attention over all $1370$ tokens, query self-attention, per-query FFN, then merged) followed by a two-layer MLP that maps to bottleneck $z\in\R^{d}$, with $d{=}512$ as the headline; $d\in\{768,1024\}$ are reported as the bottleneck-capacity ablation in Sec.~\ref{sec:e1}.

\textbf{Three diffusion decoders} are each conditioned solely on the bottleneck, $z$. The pre-trained backbones are frozen and lightweight adaptors are introduced that translate $z$ into the native conditioning format of each model (Fig.~\ref{fig:archs}b). \textbf{FM-DiT} (frozen SD3.5 MMDiT~\citep{esser2024sd3}): We introduce a $\textsc{LatentToTokens}$ adaptor using $16$ queries and $5$ QK-normed AdaLN-Zero blocks~\citep{peebles2023dit}. This generates $16{\times}4096$ tokens for the joint attention layers, while a $\textsc{Pooled Projection}$ head provides the $2048$-D global embedding. \textbf{SDXL} (frozen denoising UNet~\citep{podell2023sdxl}): The latent $z$ is mapped to $77{\times}2048$ tokens via $3$ AdaLN-Zero blocks. We initialize the outer layers with $0.10$ Xavier gain, as standard zero-initialization collapses the conditioning heads. A $\textsc{Pooled Projection}$ head maps $z$ to the $1280$-D embedding, and we apply classifier-free guidance~\citep{ho2022cfg}. \textbf{DiT} (DiT-XL/2~\citep{peebles2023dit}): Unlike the others, this model is \emph{fully trainable}, as freezing the backbone fails during the $256{\to}512$ resolution transfer. We repurpose the DiT class-embedding slot to inject $z$ using a $\textsc{ClassProj}$ head. Additionally, $16{\times}1152$ spatial tokens are injected via zero-initialized residuals at blocks $\{0,7,14,21\}$. Architectures for SDXL and DiT are deferred to App.~Fig.~\ref{fig:archs-appendix}. We also include a pixel-space VQGAN~\citep{esser2021vqgan} with a separate codebook as a non-$z$-bottleneck baseline).

\textbf{Joint training}. Encoder, conditioning adaptors, and backbone (for ViT-DiT) are trained jointly under a composite of latent-space generative terms, a latent-alignment regulariser, and a pixel-space frequency term that is gated to small denoising times $t$. The loss function per batch, with $\hat x_0$ as the clean-latent estimate, $z$ as the bottleneck, and $z'$ as its augmentation-twin embedding is given as,
\begin{equation}
  \mathcal{L} \;=\; \lambda_{\mathrm{diff}}\,\mathcal{L}_{\mathrm{diff}}
              + \lambda_{\mathrm{rec}}\,\mathcal{L}_{\mathrm{rec}}
              + \lambda_{\mathrm{con}}\,\mathcal{L}_{\mathrm{con}}
              + \lambda_{\mathrm{vic}}\,\mathcal{L}_{\mathrm{vic}}
              + \lambda_{\mathrm{freq}}\,\mathcal{L}_{\mathrm{freq}},
  \label{eq:composite-loss}
\end{equation}
where $\mathcal{L}_{\mathrm{diff}}$ is the backbone's native objective --- the rectified-flow velocity loss $\lVert v_\theta(x_t,t,z) - (\epsilon - x_0)\rVert_2^2$ for ViT-FMDiT~\citep{lipman2023flowmatching} and the standard MSE loss for ViT-SDXL and ViT-DiT; $\mathcal{L}_{\mathrm{rec}} = \mathbb{E}_t\bigl[(1{-}t)^2\,\lVert\hat x_0 - x_0\rVert_2^2\bigr]$ is a timestep-weighted clean-latent reconstruction term that emphasises low-noise steps; $\mathcal{L}_{\mathrm{con}}$ is an InfoNCE-style contrastive loss over $\{z, z'\}$ pairs; $\mathcal{L}_{\mathrm{vic}}$ is a VICReg variance--covariance regulariser~\citep{bardes2022vicreg} that keeps the bottleneck non-collapsed; $\mathcal{L}_{\mathrm{freq}} = \lVert \log(1{+}|\mathcal{F}(\hat I)|) - \log(1{+}|\mathcal{F}(I)|)\rVert_1$ is an FFT-amplitude pixel loss with a radial high-frequency mask, evaluated in on a small subsample of the batch. The five $\lambda$s are linearly ramped by a \texttt{LossWeightScheduler} (Tab.~\ref{tab:hparams-decoder}). A two-phase schedule freezes the encoder for the first seven epochs and unfreezes its last $16$ blocks at epoch $8$, in step units to absorb the topology change when the optimizer is rebuilt and the cosine schedule recomputed. More details on the implementations can be found in App. \ref{app:enc-dec}.


\subsection{Orientation codec ($\Psi$)}
\label{sec:codec}

\textbf{Why a dedicated codec?} A generic pretrained image decoder emits an RGB tensor; a crystal-plasticity solver expects an HCP orientation field. The codec is the enabling bridge $\Psi$ in Fig.~\ref{fig:hero}(b): an invertible map between HCP orientation fields and 8-bit RGB images that unlocks the decoder--optimizer loop in Sec.~\ref{sec:optimizers} to use any image-domain generator as a microstructure prior. Naively mapping Bunge--Euler orientations $(\varphi_1, \Phi, \varphi_2)$ to RGB triples corrupts the any image-domain decoder data in three ways: periodic wrap-around at $0/2\pi$ becomes ringing; For Hexagonically Closed Pack (HCP) fundamental zone, per-grain folding maps physically adjacent grains to opposite quaternions, becoming high-frequency color edges which blurs the decoder output; and recovering the $q_w$ (quaternions) produces NaNs whenever decoder noise pushes negative square-root. The codec resolves all three through a 5-step encode and a closed vectorised decode (Alg.~\ref{alg:codec}).

\textbf{Five-step pipeline}. Each per-grain orientation is converted to a unit quaternion $q\in S^3$~\citep{zhou2019} under HCP point-group $D_6$ ($|\mathcal{S}_{\mathrm{HCP}}|=12$). The encode then proceeds:
\textbf{(1)~Continuous unfolding:} breadth-first-search (BFS) over the grain-adjacency graph replaces each grain's quaternion by the symmetry equivalent closest to its already-visited parent (Eq.~\ref{eq:bfs-step}), so RGB distance between adjacent grains is proportional to their true misorientation, no artificial fundamental-zone jumps.
\textbf{(2)~Class-level anchor:} a single per-class anchor $\bar q_c$ is precomputed via the eigenvalue method of~\citet{markley2007} from $M{=}50$ file-level means; using a class-level anchor enables decoder generated images without per-sample metadata.
\textbf{(3)~Frame centring.} $q_g\leftarrow \bar q_c^{-1}\cdot q_g$ concentrates the distribution near identity, away from the $q_w{=}0$ equator where the double-cover sign flip would reintroduce discontinuities.
\textbf{(4)~Stereographic projection.} $S = q_{xyz}/(1+q_w) \in [-1,1]^3$ has the closed-form rational inverse in Eq.~\ref{eq:istereo} (no square roots, numerically stable on all of $\R^3$).
\textbf{(5)~Quantisation:} $S$ is linearly mapped to $[0, 2^b{-}1]^3$. The per-channel step $\Delta = 2/(2^b{-}1)$ bounds the worst-case angular error at ${\approx}0.78^\circ$. The 8-bit PNG variant is used inside \Co{}, it plugs directly into a pretrained $\Dec$, and its measured ${\sim}0.6^\circ$ mean error (App.~\ref{app:codec-extended}, Tab.~\ref{tab:codec-accuracy}) is far below the grain-boundary threshold~\citep{readshockley1950}. \textbf{Self-segmenting decode}is done at inference where no ground-truth label map is available for a generated image. The decoder recovers grains by linking $4$-connected pixels whose max-channel intensity differs by at most $\tau$ ($\tau{=}1$ for $8$-bit, $\tau{=}50$ for $16$-bit image) via a single sparse-graph connected-components pass~\citep{groeber2014}.

\begin{algorithm}[t]
\caption{Orientation codec --- encode (Dream3D $\to$ image) and decode (image $\to$ Dream3D)}
\label{alg:codec}
\begin{algorithmic}[1]
\Statex \textbf{Encode}\,$(\mathcal{G}, \{q_g\}, \mathcal{N}{=}(\mathcal{V},\mathcal{E}), \bar{q}_c)$
\State $g^\star \gets \arg\max_{g \in \mathcal{V}} |\{(i,j):\mathcal{G}[i,j]{=}g\}|$; $q_{g^\star} \gets \arg\max_{s,\sigma}\, \sigma\langle s\cdot q_{g^\star},\bar{q}_c\rangle$
  \Comment{root + anchor}
\For{$(g_p, g_c) \in \mathcal{E}$ from $g^\star$}
  \State $q_{g_c} \gets \arg\max_{s,\sigma}\, \sigma\langle s\cdot q_{g_c},\, q_{g_p}\rangle$
         \Comment{Eq.~\eqref{eq:bfs-step}}
\EndFor
\State $q_g \gets \bar{q}_c^{-1}\cdot q_g$ with $q_{g,w}\!\ge\!0$; $S_g \gets q_{g,xyz}/(1+q_{g,w})$
       \Comment{Eq.~\eqref{eq:stereo}}
\State \Return $\mathrm{img}[i,j] = \bigl\lfloor (S_{\mathcal{G}[i,j]}+1)/2\,\cdot\,(2^b{-}1) \bigr\rceil$
\Statex \textbf{Decode}\,$(\mathrm{img}, \bar{q}_c)$
\State $S \gets 2\,\mathrm{img}/(2^b{-}1) - 1$; $q_w \gets (1-\norm{S}^2)/(1+\norm{S}^2)$, $q_{xyz} \gets 2S/(1+\norm{S}^2)$
       \Comment{Eq.~\eqref{eq:istereo}}
\State $q \gets \bar{q}_c \cdot q$, then fold: $q \gets \arg\max_{s}\,[s\cdot q]_w$
       \Comment{HCP fundamental zone}
\State $\mathcal{G} \gets \mathrm{conn\_comp}_{4}(\mathrm{img};\,\tau)$
       \Comment{$\tau{=}1$ for $b{=}8$}
\State \Return DAMASK-ready \texttt{.dream3d} with per-pixel Euler
\end{algorithmic}
\end{algorithm}


\subsection{Crystal plasticity simulation pipeline and objectives ($f\circ\Dec$, $J$)}
\label{sec:pipeline}

With the codec in place, the materials oracle composes into
\begin{equation*}
  z \xrightarrow{\;\Dec\;} \text{PNG} \xrightarrow{\;\text{codec}\;} \text{.dream3d} \xrightarrow{\;\damask\;} \text{HDF5} \xrightarrow{\;\text{Hollomon}\;} p \xrightarrow{\;J\;} \R,
\end{equation*}
where the mechanical property tuple $p = (\sigy,\,n,\,K,\,\sigu,\,\varepsilon_u)$ collects the $0.2\%$-offset yield stress $\sigy$, the Hollomon hardening exponent $n$, strength coefficient $K$, the ultimate tensile stress $\sigu$, and the uniform strain $\varepsilon_u$ at $\sigu$. The grain table emitted by the codec (Sec.~\ref{sec:codec}) — per-grain median Euler angles, voxel counts, phase IDs — is written as a \dreamthreed{} representative volume element (RVE) polycrystal that \damask{} consumes directly. Each candidate is driven through the \damask{} \texttt{Grid} solver, an FFT-based spectral scheme for periodic RVEs~\citep{roters2019}, under uniaxial tension along the extrusion direction at a quasi-static strain rate ($\dot\varepsilon = 10^{-3}\,\mathrm{s}^{-1}$, $25\%$ total nominal strain). The constitutive law is the HCP phenomenological power law of \citet{roters2019} with the Mg-alloy AZ31-calibrated parameter set covering tensile twinning; basal, prismatic, and pyramidal slip deformation mechanisms explained in App.~\ref{app:damask}). Non-converged runs — a small minority, primarily on degenerate decoded grains — are excluded from the surrogate's training set in all optimizers.

\textbf{Objective:} $J(p)$ has two complementary forms: a \emph{toughness-weighted} strength score $J_{V1}$ that simultaneously rewards high yield $\sigy$ and high plastic work to UTS (proxied by $\sigu\,\varepsilon_u$), and a \emph{target-driven} score $J_{V2}$ that penalises weighted distance to a user-supplied target $p^\star=(\sigy^\star, n^\star, K^\star, \sigu^\star)$,
\begin{equation}
  J_{V1}(p) = \Bigl(\tfrac{\sigy}{\sigy^{\mathrm{ref}}}\Bigr)^{\!\alpha}\!\Bigl(\tfrac{\sigu\,\varepsilon_u}{T^{\mathrm{ref}}}\Bigr)^{\!\beta}
             - \lambda_n\,\mathrm{ReLU}(n_{\min}{-}n),
  \qquad
  J_{V2}(p; p^\star) = -\sqrt{\textstyle\sum_{q\in\mathcal{I}} w_q\bigl(\tfrac{q - q^\star}{|q^\star|}\bigr)^{\!2}}.
  \label{eq:objectives}
\end{equation}
The $n_{\min}$ floor in $J_{V1}$ rejects brittle solutions; both add a saturating grain-count band penalty (App.~\ref{app:damask}) since the constitutive law is grain-size insensitive and the optimizer would otherwise reach the target via a few large favourably-oriented grains or via speckle decodes with thousands of grains.

\subsection{Optimizer: \meridian{} (Ours) ($\Opt$)}
\label{sec:optimizers}
\label{sec:meridian}

We deploy \meridian{} (\emph{Manifold-Embedded Robust Inverse Design via Iterative Acquisition Networks}) as the primary optimizer and benchmark it against three latent-optimization baselines on the same box constraint $\Z = [-3,3]^{512}$.  \dante{}~\citep{wei2025} (MLP surrogate with tree exploration), \turbo{}~\citep{eriksson2019} (GP with a single trust region), and \baxus{} (GP in a $\pm 1$ random subspace)~\citep{papenmeier2022}. All implementations: the comparison table.~\ref{tab:opt-comparison} and the hyperparameter table.~\ref{tab:hparams-opt}) are in App.~\ref{app:baselines}.

\textbf{Why a dedicated optimizer?} Three properties specific to \Co{} shapes the design. The oracle is expensive ($\sim 15$\,min/sim) and a non-trivial fraction of evaluations fail at decode, codec, or solver gates, so the optimizer must model both where the response is uncertain and where it returns a finite value. Pre-trained image-domain decoders place their training mass on a thin spherical shell of the latent box (e.g.\ $\| z\| {=} 22.07{\pm}0.05$ for ViT-FMDiT-$512$), so any perturbations in $\R^{512}$ leave this shell and waste simulator calls. \dante{}'s point-estimate MLP and visit-count tree miss all three; the GP-trust-region baselines satisfy uncertainty but neither manifold geometry nor batch diversity. \meridian{} retains the iterative ``surrogate $\to$ candidate cloud $\to$ diverse batch'' skeleton common to other three optimizers and replaces every component on which it breaks for this problem (Tab.~\ref{tab:opt-comparison}).

\begin{algorithm}[t]
\caption{\meridian{}, one outer round per iteration $t = 1, \dots, T$.}
\label{alg:meridian}
\begin{algorithmic}[1]
\Require{ Box $\Z$, batch size $q$, oracle $f$, decoder $\Dec$, class pool $\mathcal P$, seed cache $\Dset_0$.}
\State $\Dset \gets \Dset_0$;\quad $L \gets L_{\mathrm{init}}$;\quad $r,\,\mathrm{plat} \gets 0$
\For{$t = 1, \dots, T$}
  \State fit trunk, feasibility head, and GP on $\Dset$ \hfill \emph{(surrogate)}
  \State $w \gets \mathrm{AS}(\nabla \hat\mu)$ if $t \ge 3$, else $\mathrm{PCA}(\mathcal P)$ \hfill \emph{(subspace)}
  \State $z_c \gets \mathrm{centroid}_{\mathrm{top}K}(\Dset)$;\quad
    $Z_{\mathrm{cand}} \gets \Pi_{\mathrm{shell}}$,\; $U \sim \mathrm{Sobol}$ \hfill \emph{(cloud)}
  \State $\alpha(z) \gets \mathrm{qLogNEI}(z;\mu,\sigma)\, g_\psi(z)$ for $V2$ \hfill \emph{(acquisition)}
  \State $S \gets \mathrm{top}_{256}(Z_{\mathrm{cand}}, \alpha)$;\quad
    $Z_{\mathrm{batch}} \gets \mathrm{GreedyDPP}(S, q;\, k_{(\phi,z)})$ \hfill \emph{(batch)}
  \State $\Dset \gets \Dset \cup \{(z, f(\Dec(z)))\}_{z \in Z_{\mathrm{batch}}}$;\quad update $L$ and $\mathrm{plat}$
  \State \textbf{if} $L < L_{\min}$ or $\mathrm{plat} \ge K_{\mathrm{plat}}$ \textbf{then} restart from $\mathcal P$ or mini-MCTS
\EndFor
\State \Return $z^\star = \arg\max_{c_i=1} y_i$ and $\Dec(z^\star)$.
\end{algorithmic}
\end{algorithm}

\textbf{One round of \meridian{}.} Each outer iteration begins by refitting the surrogate: a shared MLP feature map $\phi_\theta : \R^{512} \to \R^{16}$ is trained jointly with a logistic feasibility classifier $g_\psi$ (on \emph{all} observations, so the failure signal is preserved), after which an exact ARD-Mat\'ern-$5/2$ GP is fit on the feasible subset on top of $\phi_\theta$ to supply calibrated $(\mu, \sigma^2)$. From the surrogate we read off an importance-weighted axis vector $w \in \R^{512}$ via the active subspace of $\hat C = \E[\nabla\hat\mu\,\nabla\hat\mu^\top]$ (PCA of the class-conditional pool $\mathcal P$ during cold start)~\citep{constantine2015}. We then anchor the round on the shell-projected centroid of the top-$K$ feasible incumbents and draw a Sobol cloud around it whose perturbations are scaled by the trust-region edge $L_t$ and the importance weights $w$ on a sparse axis mask, so that proposals respect the manifold's anisotropy without ignoring rare-direction pockets, and then project the cloud onto the adaptive shell $\| z\| \in [\mu_{\| X\|} \pm k_\sigma\, \sigma_{\| X\|}]$, which is measured from the feasible data each round. Acquisition combines a feasibility-gated qLogNoisyExpectedImprovement, $\alpha_{\mathrm{GP}}(z) = \mathrm{qLogNEI}(z; \mu, \sigma)\, g_\psi(z)$, a Monte-Carlo EI term, for the target-driven $V2$ objective, computed through heteroscedastic property heads on $\phi_\theta$. The top-$256$ candidates ranked by $\alpha$ are passed to a greedy quality-weighted DPP~\citep{kulesza2012} with a hybrid $(\phi, z)$ kernel that returns a diverse batch of proposals; the batch is then evaluated through $\Dec \to \damask{}$, $L_t$ updated under success/failure cadence, and a restart fired whenever $L_t$ collapses or best plateau runs overflow $K_{\mathrm{plat}}$ rounds. \meridian{}, in this configuration with hyperparameters and more explanation are in App.~\ref{app:meridian-details}.

%==============================================================================
\section{Experiment 1: Microstructure reconstruction (E1)}
\label{sec:experiments}
\label{sec:e1}


The encoder--decoder models described in Sec.~\ref{sec:enc-dec} are trained once and then frozen for all downstream optimization experiments in Sec.~\ref{sec:e2}. Expanding on the data corpus of \citet{guru2025}, we assemble $81{,}756$ training and $9{,}084$ held-out test RGB microstructures; this expansion was achieved through geometry and physics-aware oversampling, the details of which are outside the scope of this paper and will be documented in a future publication. All the models from Tab.~\ref{tab:reconstruction} are trained on $4\times$H200 GPUs, and a standard $30$-epoch run on the training split takes roughly $12$ hours. The VQGAN reference, FM-DiT at $z\in\{512,768,1024\}$, and SDXL all use epoch-$30$ checkpoints; DiT is substantially heavier because its denoiser backbone is unfrozen, carrying about $1.17$B trainable parameters versus roughly $381$M--$391$M for the FMDiT family, and the comparison therefore uses the latest available epoch-$20$ checkpoint from the same training regime. We score reconstruction quality with three computer-vision metrics and two material science metrics. FID is the distribution-level realism score, MS-SSIM captures multi-scale structural similarity, and LPIPS captures perceptual distance~\citep{heusel2017gans,Wang2003MultiscaleSS,zhang2018perceptual}. For materials fidelity, we report $G_{\mathrm{m}}$ for grain recovery (fraction of original grains recovered at IoU $\ge 0.3$, capturing grain size and aspect-ratio agreement) and $\Delta_{\mathrm{ori}}$ for crystallography (mean disorientation between IoU-matched per grains per-class mean quaternions).

\begin{figure*}[htbp]
  \centering
  \includegraphics[width=\linewidth]{figures/Figure_reconstructions.png}
  \caption{\textbf{Qualitative Reconstructions} on AZ31 and Mg-10Gd. More examples in Fig.~\ref{fig:reconstruction-extended}}
  \label{fig:reconstruction}
\end{figure*}

\begin{table*}[htbp]
  \caption{\textbf{Reconstruction quality and microstructural fidelity on the test set} ($n = 9{,}084$).}
  \label{tab:reconstruction}
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \resizebox{0.95\textwidth}{!}{%
  \begin{tabular}{lc ccccc}
    \toprule
    \textbf{Decoder} & \textbf{$z$} & \textbf{FID $\downarrow$} & \textbf{MS-SSIM $\uparrow$} & \textbf{LPIPS $\downarrow$} & \textbf{$G_{\mathrm{m}}$ $\uparrow$} & \textbf{$\Delta_{\mathrm{ori}}$ $\downarrow$} \\
    \midrule
    \multicolumn{7}{l}{\textit{No bottleneck}} \\
    \textbf{VQGAN} & -- & 35.63 & $0.876 \pm 0.12$ & $0.132 \pm 0.10$ & $0.747 \pm 0.16$ & $27.619 \pm 10.01$ \\
    \textbf{Codec} & -- & 0.95 & $0.449 \pm 0.28$ & $0.368 \pm 0.15$ & $0.926 \pm 0.08$ & $7.200 \pm 4.26$ \\
    \midrule
    \textbf{SDXL} & 512 & 108.28 & $0.077 \pm 0.05$ & $0.650 \pm 0.08$ & $0.370 \pm 0.15$ & $56.467 \pm 8.41$ \\
    \textbf{DiT} & 512 & $\mathbf{23.19}^{*}$ & $0.147 \pm 0.18$ & $0.603 \pm 0.08$ &  $0.486 \pm 0.12$ & $55.363 \pm 10.35$ \\
    \textbf{FM-DiT} & 512 & 27.86 & $0.169 \pm 0.20$ & $0.578 \pm 0.07$ &$\mathbf{0.501 \pm 0.11}^{*}$ & $\mathbf{53.921 \pm 11.68}^{*}$ \\
    \textbf{FM-DiT} & 768 & 27.86 & $\mathbf{0.178 \pm 0.21}^{*}$ & $\mathbf{0.572 \pm 0.07}^{*}$ & $0.474 \pm 0.12$ & $54.318 \pm 11.16$ \\
    \textbf{FM-DiT} & 1024 & 27.08 & $0.174 \pm 0.21$ & $0.574 \pm 0.07$ & $0.475 \pm 0.12$ & $54.353 \pm 10.81$ \\
    \bottomrule
  \end{tabular}%
  }
  \vspace{1pt}
  \parbox{0.9\linewidth}{\footnotesize All bottlenecked decoders share the ViT based encoder; names omit the ViT prefix. $^*$Best bottlenecked decoders.}
\end{table*}


ViT-DiT achieves the best FID, while the ViT-FMDiT family (specifically at $d{=}768$) achieves the best $G_{\mathrm{m}}$—indicating the strongest grain-level recovery in size and shape—alongside the best MS-SSIM, LPIPS, and $\Delta_{\mathrm{ori}}$ among bottlenecked decoders (Tab.~\ref{tab:reconstruction}). ViT-FMDiT's performance is stable across bottleneck size, allowing the optimization loop (Sec.~\ref{sec:e2}) to use a tighter bottleneck without sacrificing fidelity. Conversely, ViT-SDXL struggles to adapt its text-conditioned UNet to a 512-D continuous bottleneck, yielding the worst metrics (FID $\sim\!108$) and visibly washed-out grains in Fig.~\ref{fig:reconstruction}. An unbottlenecked ViT-VQGAN provides an upper bound for reconstruction, while the orientation codec establishes the error floor. The codec's $\Delta_{\mathrm{ori}}$ error is safely within CP tolerances. While the bottlenecked decoders preserve macroscopic morphology and class statistics (e.g., AZ31, Mg-10Gd), their orientation space is non-isometric and symmetry-blind. This introduces a ${\sim}53^\circ$ per-pixel disorientation into the end-to-end round-trip. This residual error acts as a class-preserving reshuffle rather than a physical violation as the decoded grains remain in the HCP fundamental zone, and their texture statistics (quaternion mean, pole-figure spread) matches the target class. Therefore, we carry ViT{-}DiT and ViT{-}FMDiT with $d\in\{512,768,1024\}$ into the \Co{} loop as robust generators of statistically valid microstructures rather than exact reconstructors.

\section{Experiment 2: Materials inverse design (E2)}
\label{sec:e2}
E2 tests the central claim : Can \Co{} find statistically valid AZ31 microstructures whose simulated stress--strain summaries match a user-specified mechanical target? We use the optimizer suite from Sec.~\ref{sec:optimizers} and the trained decoders from Sec.~\ref{sec:e2}. The target, $p^\star=(\sigy^\star,n^\star,K^\star,\sigu^\star)=(160\,\mathrm{MPa},0.302,604\,\mathrm{MPa},316\,\mathrm{MPa})$ is set. Each cell uses $5$ independent runs and the same $100$ ($z_\mathbf{init}$ <-> \damask{} evaluation) seed cache for warm starting the surrogate model. The optimizer evaluates $40$ iterations with batch size $4$, i.e. $160$ new simulator evaluations per run. Average runtime per evaluation on a $700$ W H200 is $8.46$ hours with a GPU energy estimate of approx $5.93$ kWh across the full sweep. Table~\ref{tab:opt-v2} identifies ViT{-}FMDiT-$768$ with \meridian{} as the strongest target-driven configuration. After 144 evaluations (Fig.~\ref{fig:headline}), it yields the closest overall target match ($J_{V2}=-0.132\pm0.004$, $\sigy=159.1$ MPa, $n=0.299$, $\sigu=314.0$ MPa), successfully falling within the AZ31 material class. This success highlights a key principle: the optimizer and decoder must align. The $768$-D bottleneck offers the ideal trade-off—more expressive than $512$-D, yet dense enough for local surrogate modeling. \meridian{} maximizes this space through mechanisms like deep-kernel GP and feasibility heads that prevent wasted \damask{} calls; active-subspace and adaptive-shell proposals restrict batches to the latent manifold and target-aware acquisition directly aligns with $(\sigy,n,K,\sigu)$. Conversely, the $1024$-D bottleneck demonstrates that larger latents do not automatically improve inverse design. While it adds expressivity, but becomes too broad and sparse under a $160$-evaluation budget. This reinforces our core conclusion: successful inverse design requires an optimizer whose inductive biases match the encoder-decoder's latent space. This finding is further supported by the toughness-weighted $J_{V1}$ results, ablation tables, and convergence curves detailed in App.~\ref{app:extended-results}.


\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/convergence_v2_three_panel.pdf}
 \caption{\textbf{Best Mean Target-Driven $J_{V2}$} versus num. of evaluations (batch size 4, $n_{\mathrm{seed}}= 5$).}
  \label{fig:headline}
\end{figure}

\begin{figure}[!ht]
  \begin{minipage}[c]{0.3\textwidth}
    \centering
    \includegraphics[width=0.8\linewidth]{figures/optimized_v2.png}
    \captionof{figure}{Best $J_{V2}$ \\ microstructure and stress field.}
    \label{fig:placeholder-v2}
  \end{minipage}%
  \hfill
  \begin{minipage}[c]{0.7\textwidth}
    \centering
    \captionof{table}{\textbf{Optimization Results:} Target-Driven, $J_{V2}$ ($n_{\mathrm{seed}}= 5$).}
    \label{tab:opt-v2}
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{llccccc}
      \toprule
      Decoder & Optimizer & $J_{V2}\!\uparrow$ & $\sigy$ & $n$ & $\sigu$ & \# sims \\
      \midrule
      \multirow{4}{*}{ViT{-}DiT}
        & \rowcolor{gray!10} \textbf{\meridian{}}  & $-0.156 \pm 0.005^*$ & $155.9$ & $0.305$ & $313.9$ & $160$ \\
        & \dante{}               & $-0.163 \pm 0.009$ & $155.8$ & $0.305$ & $312.9$ & $104$ \\
        & \turbo{}               & $-0.168 \pm 0.007$ & $154.1$ & $0.301$ & $307.7$ & $152$ \\
        & \baxus{}               & $-0.168 \pm 0.009$ & $154.8$ & $0.303$ & $310.6$ & $152$ \\
      \midrule
      \multirow{4}{*}{ViT{-}FMDiT}
        & \rowcolor{gray!10} \textbf{\meridian{}}  & $-0.152 \pm 0.004^*$ & $156.1$ & $0.302$ & $312.1$ & $96$ \\
        & \dante{}               & $-0.161 \pm 0.016$ & $157.1$ & $0.303$ & $314.4$ & $156$ \\
        & \turbo{}               & $-0.167 \pm 0.009$ & $154.4$ & $0.301$ & $308.4$ & $160$ \\
        & \baxus{}               & $-0.162 \pm 0.005$ & $155.1$ & $0.303$ & $311.1$ & $144$ \\
      \midrule
      \multirow{4}{*}{\shortstack{ViT{-}FMDiT\\[1pt]{\footnotesize 768}}}
        & \rowcolor{blue!8} \textbf{\meridian{}}$^\dagger$  & $\mathbf{-0.132 \pm 0.004}^*$ & $159.1$ & $0.299$ & $314.0$ & $144$ \\
        & \dante{}               & $-0.140 \pm 0.004$ & $157.7$ & $0.301$ & $313.1$ & $140$ \\
        & \turbo{}               & $-0.151 \pm 0.009$ & $158.5$ & $0.306$ & $318.5$ & $148$ \\
        & \baxus{}               & $-0.146 \pm 0.004$ & $157.7$ & $0.305$ & $316.5$ & $156$ \\
      \midrule
      \multirow{4}{*}{\shortstack{ViT{-}FMDiT\\[1pt]{\footnotesize 1024}}}
        & \rowcolor{gray!10} \textbf{\meridian{}}  & $-0.150 \pm 0.005^*$ & $157.0$ & $0.304$ & $314.0$ & $152$ \\
        & \dante{}               & $-0.168 \pm 0.004$ & $153.6$ & $0.304$ & $308.8$ & $124$ \\
        & \turbo{}               & $-0.156 \pm 0.009$ & $157.1$ & $0.305$ & $315.0$ & $140$ \\
        & \baxus{}               & $-0.164 \pm 0.008$ & $155.3$ & $0.304$ & $311.6$ & $124$ \\
      \bottomrule
      \multicolumn{7}{l}{\scriptsize \colorbox{gray!10}{Grey} = our method (\meridian{}). \; $^*$ = best per decoder block. \; \colorbox{blue!8}{$^\dagger$} = best overall.}
    \end{tabular}
    \vspace{1pt}
  \end{minipage}
\end{figure}


%==============================================================================
\section{Conclusion}
\label{sec:conclusion}

\textbf{Limitations}. \Co{} is limited by its decoder as it can only search microstructures that the learned prior. The decoder is also a class-conditional generator of statistically valid AZ31 microstructures, not a faithful per-pixel orientation reconstructor; although the codec is sub-degree accurate, the encoder--decoder is still non-isometric in crystallographic rotation distance. The \damask{} oracle is HCP phenopower-law plasticity, no damage, isothermal loading, and the loading paths studied here, therefore, inverse-designed microstructures require multi-loading and experimental validation. We are addressing these issues by expanding validation and training $\Enc$--$\Dec$ with orientation-specific losses such as HCP-symmetry-aware grain-level quaternion losses and texture-statistics penalties.

\textbf{Highlights and broader impact}. \Co{} is a novel framework that turns expensive simulator-driven inverse design into active search over a learned generative design space. No similar end-to-end pipeline currently exists for this material regime. Crucially, the individual components of \Co{} are engineered to SOTA standards and have been investigated against established methods. The orientation codec acts as the bridge, making image-domain microstructure generators usable inside a crystal-plasticity loop. \meridian{} supplies the active optimizer, combining calibrated uncertainty, feasibility modeling, manifold-aware proposals, target-aware acquisition, and diverse batches to reduce wasted simulator calls. On AZ31 design, ViT-FMDiT-$768$ with \meridian{} gives the strongest target-driven result and remains effective on the toughness objective. More broadly, the same abstraction can extend to other physical systems with learned design priors and slow physics oracles.



%==============================================================================
\begin{ack}
Anonymized for double-blind review. This block is hidden during submission and revealed only in the camera-ready version.
\end{ack}


% References are managed in references.bib; numeric style via natbib option "numbers" set in preamble.
% Build sequence: pdflatex -> bibtex -> pdflatex -> pdflatex.
{\small
\bibliographystyle{unsrtnat}
\bibliography{references}
}

%==============================================================================
\appendix

%--------------------------------------------------------------------------
\section{Extended E2 results}
\label{app:extended-results}

This appendix reports the toughness-weighted objective results that complement the target-driven E2 results in Sec.~\ref{sec:e2}. The protocol is unchanged: the shared seed cache is used only for warm start, all curves and tables exclude that cache, and each run receives $160$ optimizer-phase simulator evaluations. Average wall-clock runtime for these $J_{V1}$ runs was $8.48$ hours on an H200 machine, corresponding to an estimated $5.94$ kWh of GPU energy per run under the same $700$ W TDP accounting used in the main text.

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{figures/convergence_v1_three_panel.pdf}
  \caption{\textbf{Best Mean Toughness-weighted $J_{V1}$} versus num. of evaluations (batch size 4, $n_{\mathrm{seed}}= 5$).}
  \label{fig:e2-v1-convergence}
\end{figure}

\begin{figure}[!ht]
  \begin{minipage}[c]{0.3\textwidth}
    \centering
    \includegraphics[width=0.9\linewidth]{figures/optimized_v1.png}
    \captionof{figure}{Best $J_{V2}$ \\ microstructure and stress field.}
    \label{fig:placeholder-v1}
  \end{minipage}%
  \hfill
  \begin{minipage}[c]{0.7\textwidth}
    \centering
    \captionof{table}{\textbf{Optimization Results:} Toughness-Weighted, $J_{V1}$ ($n_{\mathrm{seed}}= 5$).}
    \label{tab:opt-v1}
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \begin{tabular}{llccccc}
      \toprule
      Decoder & Optimizer & $J_{V1}\!\uparrow$ & $\sigy$ & $n$ & $\sigu$ & \# sims \\
      \midrule
      \multirow{4}{*}{ViT{-}DiT}
        & \rowcolor{gray!10} \textbf{\meridian{}}  & $0.930 \pm 0.003^*$ & $154.3$ & $0.308$ & $314.0$ & $144$ \\
        & \dante{}               & $0.929 \pm 0.010$ & $155.6$ & $0.305$ & $313.4$ & $148$ \\
        & \turbo{}               & $0.915 \pm 0.007$ & $153.8$ & $0.306$ & $311.3$ & $112$ \\
        & \baxus{}               & $0.918 \pm 0.002$ & $151.8$ & $0.312$ & $314.2$ & $160$ \\
      \midrule
      \multirow{4}{*}{ViT{-}FMDiT}
        & \rowcolor{gray!10} \textbf{\meridian{}}  & $0.943 \pm 0.011^*$ & $156.8$ & $0.306$ & $316.1$ & $124$ \\
        & \dante{}               & $0.923 \pm 0.008$ & $155.2$ & $0.305$ & $313.0$ & $152$ \\
        & \turbo{}               & $0.931 \pm 0.003$ & $154.0$ & $0.309$ & $314.2$ & $100$ \\
        & \baxus{}               & $0.933 \pm 0.006$ & $156.3$ & $0.301$ & $311.7$ & $100$ \\
      \midrule
      \multirow{4}{*}{\shortstack{ViT{-}FMDiT\\[1pt]{\footnotesize 768}}}
        & \rowcolor{blue!8} \textbf{\meridian{}}$^\dagger$  & $\mathbf{0.961 \pm 0.008}^*$ & $157.9$ & $0.305$ & $316.8$ & $112$ \\
        & \dante{}               & $0.954 \pm 0.014$ & $159.8$ & $0.300$ & $316.6$ & $140$ \\
        & \turbo{}               & $0.948 \pm 0.007$ & $156.1$ & $0.309$ & $317.9$ & $160$ \\
        & \baxus{}               & $0.953 \pm 0.004$ & $157.2$ & $0.304$ & $315.1$ & $116$ \\
      \midrule
      \multirow{4}{*}{\shortstack{ViT{-}FMDiT\\[1pt]{\footnotesize 1024}}}
        & \rowcolor{gray!10} \textbf{\meridian{}}  & $0.952 \pm 0.005^*$ & $156.2$ & $0.308$ & $317.6$ & $160$ \\
        & \dante{}               & $0.923 \pm 0.006$ & $153.5$ & $0.309$ & $314.0$ & $156$ \\
        & \turbo{}               & $0.944 \pm 0.010$ & $157.2$ & $0.305$ & $315.3$ & $140$ \\
        & \baxus{}               & $0.936 \pm 0.011$ & $155.8$ & $0.308$ & $316.7$ & $128$ \\
      \bottomrule
      \multicolumn{7}{l}{\scriptsize \colorbox{gray!10}{Grey} = our method (\meridian{}). \; $^*$ = best per decoder block. \; \colorbox{blue!8}{$^\dagger$} = best overall.}
    \end{tabular}
    \vspace{1pt}
  \end{minipage}
\end{figure}

The objective changes the design question from hitting a prescribed stress--strain tuple to finding high-strength, high-plastic-work microstructures without dropping below the hardening floor. Figure~\ref{fig:e2-v1-convergence} and Table~\ref{tab:opt-v1} show that the same decoder--optimizer pairing remains the most effective overall: ViT{-}FMDiT-$768$ with \meridian{} reaches $J_{V1}=0.961\pm0.008$, with $\sigy=157.9$ MPa, $n=0.305$, $\sigu=316.8$ MPa, and $112$ optimizer-phase simulations to the best value. Compared with the target-driven case, the best solution now moves slightly toward higher ultimate strength while preserving hardening, as expected for a toughness-weighted score.

The $1024$-D appendix rows mirror the main-text interpretation. This does not indicate a defective decoder; rather, the wider bottleneck exposes isolated texture pockets that are difficult to model globally from $160$ calls. Sparse local perturbations and low-dimensional PCA-aligned subspace moves can exploit such pockets without fitting the whole ambient latent at once, which explains why the best $J_{V1}$ rows for ViT{-}FMDiT-$1024$ come from those search biases. Across both objectives, the consistent conclusion is that ViT{-}FMDiT-$768$ gives the best operating point for \Co{}: enough latent capacity to express useful AZ31 textures, but not so much that the expensive-oracle optimizer loses sample efficiency.

%--------------------------------------------------------------------------
\section{Encoder--decoder details}
\label{app:enc-dec}

This appendix records the implementation detail behind the encoder--decoder family that the main text only sketches. The goal is not to restate Sec.~\ref{sec:enc-dec}, but to make the exact architectural, training, and validation choices reproducible: the shared encoder, the three conditioning interfaces, the composite training objective, and the bottleneck-capacity ablation.

\textbf{Shared encoder --- exact specification}. The encoder is OpenCLIP ViT-H/14 (\texttt{laion2b\_s32b\_b79k} weights)~\citep{radford2021clip,schuhmann2022laion5b}: $32$ transformer blocks, hidden width $1280$, patch size $14$, and input resolution $512{\times}512$, which yields $37{\times}37{=}1369$ patch tokens after bicubic interpolation of the original $16{\times}16$ positional embedding. We keep CLIP normalisation throughout rather than switching to ImageNet statistics. The freezing pattern is deliberately asymmetric: \texttt{conv1}, \texttt{cls\_token}, the positional embedding, \texttt{ln\_pre}, and the lower $16$ transformer blocks remain frozen, while only the upper $16$ blocks are trainable, with gradient checkpointing enabled there. This keeps most of the LAION prior intact while allowing the top of the trunk to adapt to EBSD imagery. The CLS readout is replaced by a multi-query attention pooler with $4$ learned queries (initialised $\mathcal{N}(0,0.02)$), $8$ attention heads, query self-attention, and a per-query FFN, followed by a merge head $\mathrm{Linear}(4D \to D)$. The bottleneck MLP is $\mathrm{Linear}(1280,1280)\to\mathrm{GELU}\to\mathrm{Linear}(1280,d)\to\mathrm{LayerNorm}$ with $d \in \{512,768,1024\}$.

\textbf{Conditioning adaptors --- exact specifications}. All three decoders reuse the same bottleneck but expose different native conditioning interfaces, so each adaptor is designed to match the backbone it plugs into rather than to force a uniform abstraction.
\textbf{FM-DiT.} $\textsc{LatentToTokens}(d \to 16{\times}4096)$ uses $16$ learned queries, $5$ AdaLN-Zero blocks, and QK-normalised cross-attention against the pooled latent. Its outer $\mathrm{proj\_out}$ is zero-initialised, which is stable here because flow matching learns a residual velocity field. The pooled head is $\mathrm{Linear}(d,2048)\to\mathrm{SiLU}\to\mathrm{Linear}(2048,2048)$, again with a zero-initialised last layer. The adaptor carries roughly $11$M trainable parameters and the pooled head another $4$M. The frozen backbone is \texttt{stabilityai/stable-diffusion-3.5-medium}; its VAE uses $16$ channels, $8{\times}$ downsampling, and scale factor $1.5305$. Timesteps are sampled as $t = \sigma(\mathcal{N}(0,1)+\log\,\mathrm{shift})$ with $\mathrm{shift}{=}3.0$, matching the published SD3.5 logit-normal schedule, and inference uses FlowMatchEulerDiscrete.
\textbf{SDXL.} $\textsc{LatentToTokens}(d \to 77{\times}2048)$ mirrors the original SDXL text sequence length with $77$ learned queries passed through $3$ AdaLN-Zero blocks. Here zero-initialisation was empirically unstable, so both the outer projection and the pooled head are initialised with Xavier gain $0.10$; runs with zero init produced dead conditioning heads (see \texttt{samples/vitsdxl/run\_20260426\_*}). The pooled branch is $\mathrm{Linear}(d,1280)\to\mathrm{SiLU}\to\mathrm{Linear}(1280,1280)$ feeding \texttt{add\_embedding}, and a learned $\textsc{NullToken}$ implements classifier-free guidance dropout with $p_{\mathrm{drop}}{=}0.1$. The trainable interface is roughly $8$M parameters in the token adaptor, $0.8$M in the pooled head, plus the $512$-parameter null token. The SDXL UNet backbone remains frozen in bf16; training uses DDPM and inference EulerDiscrete.
\textbf{DiT.} DiT-XL/2 (hidden size $1152$, $28$ blocks, $16$ heads, patch size $2$, VAE \texttt{sd-vae-ft-mse}) is fully trainable because freezing failed under the $256{\to}512$ transfer. The class-conditioning slot is repurposed through $\mathrm{ClassProj}(d \to 1152) = \mathrm{Linear}\to\mathrm{GELU}\to\mathrm{Linear}$ with Xavier initialisation. A pair of forward pre-hooks (\texttt{\_CLIPInjector}/\texttt{\_ZeroEmbedder}) removes the original repeated class-embedding path and replaces it with a single top-of-stack injection of $\mathrm{ClassProj}(z)$. In parallel, $\textsc{LatentToSpatialTokens}(d \to 16{\times}1152)$ emits $16$ spatial tokens that a $\textsc{SpatialCrossFuser}$ inserts, with zero-initialised output projection, at blocks $\{0,7,14,21\}$. Training uses DDPM with $1000$ steps and linearly spaced $\beta$ from $10^{-4}$ to $0.02$; inference uses $250$-step DDIM.

\begin{figure}[!ht]
  \centering
  \includegraphics[width=\linewidth]{figures/Figure_enc_dec_appendix.png}
  \caption{\textbf{Companion decoder pipelines} (deferred from Fig.~\ref{fig:archs}). (a)~ViT$+$DiT-XL/2 (\emph{trainable}, $256{\to}512$): $\textsc{ClassProj}$ repurposes the DiT class-embedding slot to inject $z$; $16{\times}1152$ spatial tokens; SD~1.5 VAE. (b)~ViT$+$SDXL (frozen): $77{\times}2048$ tokens via $3$ AdaLN-Zero blocks (Xavier-$0.10$) plus $\textsc{Pooled Projection}$ $\to$ $1280$-D; SD-XL VAE.}
  \label{fig:archs-appendix}
\end{figure}

\textbf{Composite loss and training schedule}. Training combines the five terms introduced in Eq.~\eqref{eq:composite-loss} rather than relying on the backbone loss alone. The diffusion term $\mathcal{L}_{\mathrm{diff}}$ is the rectified-flow velocity MSE $\lVert v_\theta(x_t,t,z) - (\epsilon - x_0)\rVert_2^2$ for ViT-FMDiT and the standard $\epsilon$-prediction MSE $\lVert\epsilon_\theta(x_t,t,z) - \epsilon\rVert_2^2$ for SDXL and ViT-DiT, always evaluated in the native VAE latent of the corresponding backbone. The reconstruction term is $\mathcal{L}_{\mathrm{rec}} = \mathbb{E}_t[(1{-}t)^2\lVert\hat x_0-x_0\rVert_2^2]$, which biases training toward cleaner timesteps; for the velocity parameterisation, $\hat x_0 = x_t - t v_\theta$. On top of this, we add a contrastive InfoNCE-style term $\mathcal{L}_{\mathrm{con}}$ on augmentation twins $(z,z')$, a VICReg term $\mathcal{L}_{\mathrm{vic}}$~\citep{bardes2022vicreg} with cross-rank all-gather ($\textsc{var\_w}{=}25$, $\textsc{cov\_w}{=}1$, target std $1$), and a frequency-domain loss $\mathcal{L}_{\mathrm{freq}}$ given by an $L_1$ distance between radially weighted log-amplitude FFT spectra of the decoded prediction and the target image. The FFT term is capped to at most two image pairs per batch and only evaluated for $t<0.7$ so that it improves high-frequency fidelity without destabilising high-noise updates.

The outer weights follow a scheduled ramp rather than appearing at full strength immediately: at the end of the ramp the defaults are $(\lambda_{\mathrm{diff}},\lambda_{\mathrm{rec}},\lambda_{\mathrm{con}},\lambda_{\mathrm{vic}},\lambda_{\mathrm{freq}})=(1.0,0.5,0.1,0.05,0.30)$, with onset and ramp length controlled by the \texttt{LossWeightScheduler} in Tab.~\ref{tab:hparams-decoder}. We use a two-phase encoder schedule: epochs $1$--$7$ update only the adaptors and bottleneck, then epoch $8$ unfreezes the upper $16$ ViT blocks, rebuilds the optimizer, and recomputes the cosine schedule in optimizer-step units. Training runs under PyTorch DDP on $4{\times}$H200 NVL with \texttt{find\_unused\_parameters=True} to tolerate the topology change; a per-rank skip handshake ensures that NaN-guarded steps are either taken or skipped synchronously across ranks.

\textbf{Bottleneck-capacity ablation ($d\in\{512,768,1024\}$)}. Bottleneck width is the most controlled ablation in this family because it leaves the decoder, encoder, conditioning path, and loss schedule unchanged. Under that constraint, held-out ViT-FMDiT PSNR improves monotonically with width, from roughly $14.0$ dB at epoch $2$ for $d{=}768$ to roughly $16.9$ dB at epoch $2$ for $d{=}1024$. This is consistent with the main-text choice of $d{=}512$ as a deliberately severe bottleneck for the latent-BO experiments rather than as the reconstruction-optimal point. Final-epoch reconstruction metrics for all three widths appear in Tab.~\ref{tab:reconstruction}.


\begin{table}[!ht]
  \caption{Decoder-specific hyperparameters on $4{\times}$H200. Batch size $=$ per-GPU $\times$ accum $\times$ \#GPUs.}
  \label{tab:hparams-decoder}
  \centering
  \footnotesize
  \setlength{\tabcolsep}{3pt}
  \begin{tabularx}{\linewidth}{@{}l X X X@{}}
    \toprule
    Hyperparameter & ViT-FMDiT & ViT-DiT & ViT-SDXL \\
    \midrule
    Backbone                          & SD3.5 MMDiT (${\sim}2.5$\,B) & DiT-XL/2 (${\sim}0.7$\,B) & SDXL UNet (${\sim}2.6$\,B) \\
    Backbone state                    & frozen        & \textbf{trainable}      & frozen \,(LoRA $r{=}16$) \\
    VAE                               & SD3.5 (16ch, $s{=}1.5305$) & sd-vae-ft-mse (4ch) & SD-XL VAE \\
    Bottleneck $d$ (head./abl.)       & $512/\{768,1024\}$ & $512/\{768,1024\}$ & $512$ \\
    LatentToTokens (\#q / depth)      & $16 / 5$      & $16$ spatial / --        & $77 / 3$ \\
    Conditioning dim                  & $4096$        & $1152$                   & $2048$ \\
    Pooled-projection dim             & $2048$        & ---                      & $1280$ \\
    Spatial injection blocks          & ---           & $\{0,7,14,21\}$          & --- \\
    Outer-proj init                   & zero          & Xavier ($0.10$)          & Xavier ($0.10$) \\
    Trainable interface               & ${\sim}0.6\%$ & $100\%$                  & ${\sim}0.4\%$ \\
    \midrule
    Per-GPU/accum/eff.\ batch         & $48/3/576$    & $144/1/576$              & $144/2/1152$ \\
    Train scheduler                   & FM-Euler      & DDPM (lin., $1000$)      & DDPM \\
    Inference scheduler ($N$ steps)   & FM-Euler ($50$) & DDIM ($250$)           & EulerDiscrete \\
    Timestep distribution             & logit-N, shift $3$ & uniform             & uniform \\
    CFG dropout                       & ---           & ---                      & $p_{\mathrm{drop}}{=}0.1$ \\
    \bottomrule
  \end{tabularx}
\end{table}


\begin{figure}[!ht]
  \centering
  \includegraphics[width=\linewidth]{figures/Figure_reconstructions_appendix.png}
  \caption{\textbf{Extended reconstruction gallery on five additional alloy classes} (rows, in order: \texttt{ME21}, \texttt{ZX10}, \texttt{Mg-5Gd-0.5Mn}, \texttt{Z1}, \texttt{ZNd10}); columns follow Fig.~\ref{fig:reconstruction}.}
  \label{fig:reconstruction-extended}
\end{figure}

%--------------------------------------------------------------------------
\section{Orientation codec --- extended theory and round-trip diagnostics}
\label{app:codec-extended}

This section expands the codec analysis from Sec.~\ref{sec:codec} and makes explicit the geometric choices that let a generic image decoder serve as a microstructure prior. The thread is simple: every step is chosen to preserve local orientation structure while keeping the RGB image numerically stable for downstream generative models.

\textbf{Quaternion parameterisation and HCP symmetry}. We first convert each per-grain orientation from Bunge--Euler angles ($ZXZ$) to a unit quaternion $q=(q_x,q_y,q_z,q_w)\in S^3$~\citep{zhou2019}. Because quaternions form a double cover ($q\equiv -q$), we fix the sign by enforcing the hemisphere convention $q_w\ge 0$. The relevant crystal symmetry is the HCP point group $D_6$ with $|\mathcal{S}_{\mathrm{HCP}}|=12$, generated by the six $c$-axis rotations $R_z(k\pi/3)$, $k=0,\ldots,5$, and the six compositions $R_z(k\pi/3)\cdot R_x(\pi)$. In quaternion coordinates $R_z(\theta)=(\sin(\theta/2)\,\hat z,\cos(\theta/2))$ under the standard Hamilton product. Two orientations are therefore physically identical whenever they differ by a symmetry element $s_k \in \mathcal{S}_{\mathrm{HCP}}$.

\textbf{Step 1 --- continuous unfolding (anchored BFS)}. Let $\mathcal{N}=(\mathcal{V},\mathcal{E})$ denote the grain adjacency graph from \dreamthreed{}, with $|\mathcal{V}|=G$ grains and $\mathcal{E}$ the set of $4$-connected boundary pairs. We choose the BFS root deterministically as the largest grain,
\begin{equation}
  g^\star = \arg\max_{g \in \mathcal{V}}\; \bigl|\{(i,j) : \mathcal{G}[i,j] = g\}\bigr|,
  \label{eq:bfs-root}
\end{equation}
and align its quaternion to the class anchor (Step 2) by $q_{g^\star} \leftarrow \arg\max_{s,\sigma}\, \sigma\,\langle s\cdot q_{g^\star},\, \bar{q}_c\rangle$ over $s\in\mathcal{S}_{\mathrm{HCP}}$ and $\sigma\in\{\pm 1\}$, where $\langle\cdot,\cdot\rangle$ is the $\R^4$ inner product. Using the largest grain as the seed shortens the average BFS path length, and aligning that single seed to $\bar q_c$ keeps the unfolded tree on a common hemisphere so that identical orientations map to consistent colours across the class. Each remaining grain then inherits, from its already visited neighbour $g_p$, the symmetry equivalent and sign with maximal inner product:
\begin{equation}
  q_{g_c} \;\leftarrow\; \arg\max_{s,\sigma}\; \sigma\, \langle s \cdot q_{g_c},\; q_{g_p} \rangle.
  \label{eq:bfs-step}
\end{equation}
Each edge requires $24$ dot products, so the encode remains $\mathcal{O}(|\mathcal{E}|)$ in the boundary count. After this pass, quaternion distance between adjacent grains tracks their physical misorientation directly, without the artificial jumps induced by per-grain folding to a fundamental zone.

\textbf{Step 2 --- class anchor via Markley averaging}. For each class $c$, the anchor $\bar q_c$ is the $L_2$-optimal mean of $M{=}50$ representative file-level mean quaternions $\{m_j\}_{j=1}^M$. Following \citet{markley2007}, it is the leading eigenvector of
\begin{equation}
  A = \frac{1}{M}\sum_{j=1}^{M} m_j\, m_j^{\top} \;\in\; \R^{4\times 4}.
  \label{eq:markley}
\end{equation}
The point of using a class-level rather than per-sample anchor is that generated images arrive without per-sample metadata. The cost is a mild long-range colour drift induced by within-class spread, but at $M{=}50$ that drift is still comfortably inside the capacity of the encoder bottleneck.

\textbf{Steps 3--4 --- frame centring and stereographic projection}. Once the anchor is factored out via $q_g \leftarrow \bar q_c^{-1}\cdot q_g$, the distribution concentrates near identity ($q_w\approx 1$, $\norm{q_{xyz}}\ll 1$). Enforcing $q_w\ge 0$, we project to
\begin{equation}
  S \;=\; \frac{(q_x,\, q_y,\, q_z)}{1 + q_w} \;\in\; [-1, 1]^3,
  \label{eq:stereo}
\end{equation}
which is $C^\infty$ on the open hemisphere $q_w>0$. The inverse map is closed form,
\begin{equation}
  q_w \;=\; \frac{1 - \norm{S}^2}{1 + \norm{S}^2}, \qquad
  (q_x, q_y, q_z) \;=\; \frac{2\, S}{1 + \norm{S}^2},
  \label{eq:istereo}
\end{equation}
and uses only rational arithmetic. This is the crucial numerical simplification: there is no square root, so decoder noise cannot drive the inverse into NaNs. As $\norm{S}\to\infty$, $q_w$ approaches $-1$ smoothly rather than catastrophically.

\textbf{Step 5 --- quantisation error bound}. For a centred unit quaternion near identity, let $S=q_{xyz}/(1+q_w)$ and perturb it by $\delta S$. To first order, $\delta q_{xyz}\approx (1+q_w)\,\delta S$ while $\delta q_w$ is second order. The induced angular error is therefore $\delta\theta = 2\arcsin(\norm{\delta q_{xyz}}) \approx 2(1+q_w)\norm{\delta S} \approx 2\norm{\delta S}$ near identity. With quantisation step $\Delta = 2/(2^b{-}1)$, this yields the worst-case bound $\delta\theta \lesssim 0.78^\circ$ for $b{=}8$ and $0.004^\circ$ for $b{=}16$. The measured PNG mean in Tab.~\ref{tab:codec-accuracy} is $0.6^\circ$, in line with that estimate.

\begin{table}[!ht]
  \caption{Codec round-trip accuracy on three Mg-alloy classes ($300{\times}300$ images, $257$--$2{,}044$ grains/sample). Boundary F1 and grain-size Pearson correlation are unchanged between TIFF and PNG; only quantisation noise differs.}
  \label{tab:codec-accuracy}
  \centering
  \small
  \begin{tabular}{lcccc}
    \toprule
    Class & Boundary F1 & Grain-size $r$ & PNG-vs-TIFF mean ($^\circ$) & PNG-vs-TIFF max ($^\circ$) \\
    \midrule
    \texttt{ME21\_extruded}    & $0.9999$ & $0.9997$ & $0.61$ & $1.35$ \\
    \texttt{Mg{-}10Gd\_extruded} & $1.0000$ & $0.9993$ & $0.69$ & $1.30$ \\
    \texttt{AZ31\_extruded\_HT} & $0.9999$ & $0.9991$ & $0.61$ & $1.37$ \\
    \bottomrule
  \end{tabular}
\end{table}

\textbf{Why these choices}. The design logic is now clear. The rational inverse in \eqref{eq:istereo} replaces the unstable recovery $\sqrt{1-q_x^2-q_y^2-q_z^2}$, which fails as soon as decoder noise makes the radicand negative. Continuous unfolding preserves the proportionality between RGB distance and true misorientation, keeping the encoded image in the low-frequency regime that pretrained image backbones model well. The class-level Markley anchor is not mathematically ideal, but it is the only option compatible with generated images that lack per-sample metadata. Finally, we use $b{=}8$ inside \Co{} because pretrained ViT and diffusion backbones expect 8-bit RGB, and the resulting $\sim 0.6^\circ$ floor remains below both typical EBSD angular resolution and the Read--Shockley low-angle threshold.

%--------------------------------------------------------------------------
\section{DAMASK and the HCP phenopower-law constitutive model ($\mathbf{F}_p$, $\dot\gamma^\alpha$, $\xi^\alpha$)}
\label{app:damask}

This section records the constitutive model and post-processing pipeline actually used by the oracle. The main text only needs the fact that \damask{} returns a stress--strain response; here we spell out the kinematics, hardening law, and property extraction so the simulation layer is unambiguous.

\textbf{Kinematics and resolved shear stress}. We use the standard multiplicative split $\mathbf{F}=\mathbf{F}_e\mathbf{F}_p$, which separates elastic lattice stretch from plastic shear~\citep{Wang2021}. Plastic flow is written as the sum over all active slip and twin systems,
$\mathbf{L}_p = \sum_\alpha \dot\gamma^\alpha\, \mathbf{s}^\alpha \otimes \mathbf{m}^\alpha$,
with $\mathbf{s}^\alpha$ and $\mathbf{m}^\alpha$ the unit slip or twin direction and plane normal. The resolved shear stress on system $\alpha$ is then $\tau^\alpha = \boldsymbol{\sigma}\!:\!(\mathbf{s}^\alpha\otimes\mathbf{m}^\alpha)$.

\textbf{Power-law flow rule}. Shear rates follow a phenomenological power law, separately on slip ($sl$) and twin ($tw$) systems:
\begin{equation}
  \dot\gamma^\alpha_{sl} = \dot\gamma_0^{sl}\,\Bigl|\tfrac{\tau^\alpha}{\xi^\alpha_{sl}}\Bigr|^{n_{sl}}\!\mathrm{sign}(\tau^\alpha),
  \qquad
  \dot\gamma^\alpha_{tw} = \dot\gamma_0^{tw}\,\Bigl(\tfrac{\tau^\alpha}{\xi^\alpha_{tw}}\Bigr)^{n_{tw}},\;\;\tau^\alpha > 0,
  \label{eq:phenopower-flow}
\end{equation}
with reference rates $\dot\gamma_0^{sl/tw}$, rate-sensitivity exponents $n_{sl/tw}$, and critical resolved shear stresses $\xi^\alpha_{sl/tw}$. Twinning contributes only for $\tau^\alpha>0$, reflecting the polarity of twin shear in HCP crystals~\citep{ActaMg2014}.

\textbf{Hardening evolution}. The CRSS evolves through self- and latent-hardening interactions,
$\dot\xi^\alpha = \sum_\beta h^{\alpha\beta}\,|\dot\gamma^\beta|$,
where $h^{\alpha\beta}$ is the system-interaction matrix. Slip systems use saturation-type hardening with explicit slip--twin coupling,
\begin{equation}
  \dot\xi^\alpha_{sl} = h_0^{sl}\,\bigl(1 - \xi^\alpha_{sl}/\xi^\infty_{sl}\bigr)^{a_{sl}}\!\sum_\beta h^{\alpha\beta}_{sl\text{-}sl}\,|\dot\gamma^\beta_{sl}|
     + f^{sl\text{-}tw}_{sat}\!\sum_\beta h^{\alpha\beta}_{sl\text{-}tw}\,|\dot\gamma^\beta_{tw}|,
  \label{eq:phenopower-slip-hardening}
\end{equation}
with initial slip-hardening modulus $h_0^{sl}$, saturation CRSS $\xi^\infty_{sl}$, hardening exponent $a_{sl}$, and slip--twin coupling factor $f^{sl\text{-}tw}_{sat}$. Twin systems harden symmetrically through twin--twin and twin--slip channels,
\begin{equation}
  \dot\xi^\alpha_{tw} = h_0^{tw\text{-}tw}\!\sum_\beta h^{\alpha\beta}_{tw\text{-}tw}\,|\dot\gamma^\beta_{tw}|
     + h_0^{tw\text{-}sl}\!\sum_\beta h^{\alpha\beta}_{tw\text{-}sl}\,|\dot\gamma^\beta_{sl}|.
  \label{eq:phenopower-twin-hardening}
\end{equation}
The model is intentionally phenomenological: hardening is encoded through these scalar moduli rather than through dislocation-density or twin-volume-fraction state variables~\citep{Wang2021}. That choice keeps the oracle tractable while preserving the slip--twin asymmetry that matters most in HCP magnesium~\citep{ActaMg2014}. Although damage-augmented variants exist~\citep{FeCrAl2021}, we do not use them here. Instead, every run uses the AZ31-calibrated parameter file distributed with \damask{} (\texttt{AZ31\_Phenopower.yaml}), with basal, prismatic, and pyramidal-$\langle a\rangle$/-$\langle c{+}a\rangle$ slip plus tensile twinning, under isothermal uniaxial tension along the extrusion direction and traction-free transverse faces.

\textbf{Hollomon fit and property extraction}. Post-processing is straightforward but consistent across all runs. We read the HDF5 output, compute volume-averaged Cauchy stress and logarithmic strain along the loading axis at every increment, identify the elastic--plastic transition by the $0.2\%$ offset rule, and fit the plastic branch in log--log coordinates with the Hollomon law $\sigma = K\varepsilon_p^n$. The yield stress $\sigy$ is the offset intercept, $\sigu$ the maximum stress, and $(K,n)$ the fitted Hollomon parameters. Runs that do not converge, contain NaN stresses, or produce fewer than $50$ converged increments are discarded and logged.

\begin{table}[!ht]
  \caption{\damask{} simulation parameters used for all property evaluations. The same \texttt{load.yaml} and \texttt{AZ31\_Phenopower.yaml} are used across every cell of E2; only the input \texttt{.dream3d} (i.e., the decoded microstructure) varies.}
  \label{tab:hparams-sim}
  \centering
  \small
  \begin{tabular}{ll}
    \toprule
    Parameter & Value \\
    \midrule
    Loading mode                  & uniaxial tension along ED \\
    Total time $t$ (s)            & $250$ \\
    Increments $N$                & $1250$ \\
    Strain rate $\dot{\varepsilon}$ (s$^{-1}$) & $1.0 \times 10^{-3}$ \\
    Output frequency $f_{\mathrm{out}}$ & every increment \\
    Constitutive law              & phenopowerlaw (HCP) \\
    Slip families                 & basal, prismatic, pyr.\ $\langle a\rangle$, pyr.\ $\langle c{+}a\rangle$ \\
    Twinning                      & tensile twin only \\
    Phase parameter file          & \texttt{AZ31\_Phenopower.yaml} \\
    Failure threshold             & ${<}50$ converged increments $\Rightarrow$ drop \\
    \bottomrule
  \end{tabular}
\end{table}

%--------------------------------------------------------------------------
\section{Baseline optimizer implementations and hyperparameters}
\label{app:baselines}

This appendix gathers the implementation detail for the optimizer benchmark that would be distracting in the main text: the pipeline-level adaptations applied uniformly across methods, the concrete baseline settings, the compact design comparison in Tab.~\ref{tab:opt-comparison}, and the full hyperparameter table in Tab.~\ref{tab:hparams-opt}. All values are those committed in \texttt{config/az31\_*.yaml} and \texttt{copilot/optimizers/}; no per-decoder, per-objective, or per-seed retuning was introduced after the sweep design was fixed.

\subsection{Pipeline-level adaptations applied to all four optimizers}

\textbf{Surrogate masking of failed evaluations.}
The decode$\to$codec$\to$\damask{} pipeline can reject a candidate for several reasons: the decoded image may collapse to blank or speckle structure, it may fail the chromatic-spread gate, it may produce fewer than $g_{\min}{=}40$ grains, or the simulator may fail to converge. Following standard latent-BO practice~\citep{eriksson2019,papenmeier2022}, we log such cases for bookkeeping but exclude them from the regression surrogate in all four optimizers. A constant penalty would pollute posterior variance without adding useful structure and would do so symmetrically across baselines. \meridian{} is the only method that still retains these failures explicitly through its feasibility classifier $g_\psi$.

\textbf{Trust-region restart cap.}
The second global intervention is the restart cap. Under the published \turbo{} rule $\tau_{\mathrm{fail}}=\lceil d/q \cdot \alpha \rceil$, the failure tolerance becomes $192$ for \turbo{} and \meridian{} at $(d,q,\alpha)=(512,4,1.5)$, and $128$ for \baxus{} at its initial subspace dimension $d_0=128$. In a $200$-evaluation regime that effectively disables restarts and turns all three methods into one-shot local searches. We therefore impose a shared guardrail $\tau_{\mathrm{fail}}^{\max}=8$ on the three trust-region methods while leaving the published $\alpha$ unchanged. The point is not to strengthen any one method, but to make the documented restart logic reachable at all. \dante{} has no trust-region mechanism and is unchanged.

\subsection{\dante{}~\citep{wei2025}}

For \dante{}, we use the authors' Neural Tree Explorer (NTE): an iterative root-and-cloud DUCB tree search paired with a deep MLP surrogate. The surrogate has hidden widths $[1024,512,256]$, dropout $p=0.1$, and is trained for $200$ epochs with AdamW (lr $10^{-3}$, weight decay $10^{-4}$, batch size $32$, validation split $0.1$, early-stopping patience $20$). Each round expands $200$ leaves with branching factor $16$ and depth-decayed Gaussian leaves using $\sigma_{\mathrm{init}}=0.05$, $\sigma_{\mathrm{decay}}=0.995$, DUCB constant $c_0=0.1$, and smoothing $\rho=0.5$. Since \dante{} has no native restart mechanism, we leave its visit-accumulation behaviour untouched.

\subsection{\turbo{}~\citep{eriksson2019}}

\turbo{} is run as a single trust-region GP method with an ARD-Mat\'ern-$5/2$ kernel and Thompson sampling over $5{,}000$ Sobol candidates. The success/failure cadence is the standard one, $\tau_{\mathrm{succ}}{=}3$ and $\tau_{\mathrm{fail}}{=}\lceil d/q \cdot \alpha\rceil$ with $\alpha=1.5$, but with the shared cap $\tau_{\mathrm{fail}}^{\max}{=}8$ described above. Trust-region bounds are $L\in[0.005,1.6]$ with $L_{\mathrm{init}}=0.8$, and collapse triggers a Sobol cold start in $\Z$. We also activate a plateau-restart guardrail (B1): after $K_{\mathrm{plat}}{=}5$ non-improving batches, the method is restarted even if the trust region has not yet formally collapsed. Without this extra trigger, the published shrink schedule would require about seven halvings to reach $L_{\min}$, which does not fit inside our $40$-iteration budget.

\subsection{\baxus{}~\citep{papenmeier2022}}

\baxus{} uses the same acquisition and trust-region cadence as \turbo{}, but fits its GP in a low-dimensional embedded subspace. We begin at the authors' default target dimension $d_0=10$ and replace the usual random sparse $\pm 1$ embedding with a PCA-of-seeds basis (B2): the projector columns are the leading PCA directions of the class-conditional latent pool $\mathcal P$, truncated to $k_{\mathrm{eff}}=\min(d_0,\mathrm{rank}(\mathcal P))$. This keeps lifted proposals aligned with the encoded seed support. We also enable reject feedback to the GP (\texttt{wants\_reject\_feedback = True}) so that soft-floor codec failures are treated as informative evidence about bad directions rather than discarded.

\begin{table}[!ht]
  \caption{Key design distinctions among the four latent-optimizers benchmarked in this work. Mechanisms unique to \meridian{} are highlighted in bold.}
  \label{tab:opt-comparison}
  \centering
  \small
  \setlength{\tabcolsep}{4pt}
  \begin{tabularx}{\linewidth}{l XXXX}
    \toprule
    Design axis & \dante{}~\citep{wei2025} & \turbo{}~\citep{eriksson2019} & \baxus{}~\citep{papenmeier2022} & \meridian{} (ours) \\
    \midrule
    Surrogate
      & MLP point estimate
      & ARD-Mat\'ern-$5/2$ GP
      & ARD-Mat\'ern-$5/2$ GP in subspace
      & \textbf{deep-kernel GP} \\
    Target-aware signal
      & scalar-$Y$ only
      & scalar-$Y$ only
      & scalar-$Y$ only
      & \textbf{property-head EI for $V2$} \\
    Feasibility handling
      & none
      & none
      & reject feedback to GP
      & \textbf{shared-trunk classifier} $g_\psi$ \\
    Proposal geometry
      & global tree expansion
      & isotropic trust region
      & low-dim subspace trust region
      & \textbf{anisotropic trust region} \\
    Manifold prior
      & none
      & none
      & PCA-of-seeds embedding
      & \textbf{active subspace + adaptive shell} \\
    Batch strategy
      & top-$k$ predicted
      & max-posterior sampling
      & max-posterior sampling
      & \textbf{quality-weighted DPP} \\
    Restart rule
      & none
      & Sobol restart + plateau cap
      & subspace doubling
      & \textbf{class-pool or mini-MCTS restart} \\
    Final-phase behavior
      & monotone
      & monotone
      & monotone
      & \textbf{explicit polish phase} \\
    \bottomrule
  \end{tabularx}
\end{table}


\subsection{\meridian{} (ours)}
\label{app:meridian-details}


\begin{table*}[t]
  \caption{Optimizer hyperparameters for all four optimizers share batch size $q=4$, latent dimension $d=512$, search box $\Z=[-3,3]^{512}$, $T=40$ outer iterations, and seed-cache warm start $n_0=100$.}
  \label{tab:hparams-opt}
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \renewcommand{\arraystretch}{1.04}
  \begin{minipage}[t]{0.48\textwidth}
    \centering
    \textbf{\dante{}}\\[1pt]
    \begin{tabular}{@{}p{0.38\linewidth}p{0.56\linewidth}@{}}
      \toprule
      Model & MLP $[1024,512,256]$ \\
      Acquisition & DUCB \\
      Candidates & $200$ leaves \\
      Geometry & branching $16$ tree \\
      Training & AdamW, $200$ epochs \\
      lr / wd & $10^{-3}$ / $10^{-4}$ \\
      Regularisation & dropout $0.1$ \\
      Exploration & $\sigma_{\mathrm{init}}=0.05$ \\
      Decay & $\sigma_{\mathrm{decay}}=0.995$ \\
      Score & $c_0=0.1$, $\rho=0.5$ \\
      Restart & none \\
      \\
      \bottomrule
    \end{tabular}
  \end{minipage}\hfill
  \begin{minipage}[t]{0.48\textwidth}
    \centering
    \textbf{\turbo{}}\\[1pt]
    \begin{tabular}{@{}p{0.38\linewidth}p{0.56\linewidth}@{}}
      \toprule
      Surrogate & ARD-Mat\'ern-$5/2$ GP \\
      Acquisition & Thompson \\
      Candidates & $5{,}000$ Sobol \\
      Geometry & isotropic trust region \\
      Trust region & $L_{\mathrm{init}}=0.8$, $L_{\min}=0.005$, $L_{\max}=1.6$ \\
      Success rule & $\tau_{\mathrm{succ}}=3$ \\
      Failure rule & $\alpha=1.5$, $\tau_{\mathrm{fail}}^{\max}=8$ \\
      Plateau cap & $K_{\mathrm{plat}}=5$ \\
      Warm start & Sobol cache \\
      Restart & Sobol cold-start \\
      Polish & --- \\
      \bottomrule
    \end{tabular}
  \end{minipage}

  \vspace{4pt}

  \begin{minipage}[t]{0.48\textwidth}
    \centering
    \textbf{\baxus{}}\\[1pt]
    \begin{tabular}{@{}p{0.38\linewidth}p{0.56\linewidth}@{}}
      \toprule
      Surrogate & ARD-Mat\'ern-$5/2$ GP \\
      Acquisition & Thompson \\
      Candidates & $5{,}000$ Sobol \\
      Geometry & subspace trust region \\
      Subspace & PCA-of-seeds, $d_0=10$ \\
      Warm start & PCA of $\mathcal{P}$ \\
      Trust region & $L_{\mathrm{init}}=0.8$, $L_{\min}=0.005$, $L_{\max}=1.6$ \\
      Success rule & $\tau_{\mathrm{succ}}=3$ \\
      Failure rule & $\alpha=1.5$, $\tau_{\mathrm{fail}}^{\max}=8$ \\
      Reject feedback & on \\
      Restart & subspace doubling \\
      Polish & --- \\
      \\
      \bottomrule
    \end{tabular}
  \end{minipage}\hfill
  \begin{minipage}[t]{0.48\textwidth}
    \centering
    \textbf{\meridian{}}\\[1pt]
    \begin{tabular}{@{}p{0.38\linewidth}p{0.56\linewidth}@{}}
      \toprule
      Trunk & MLP $[512,128]\to 16$ \\
      Heads & feasibility + property ($V2$) \\
      Acquisition & qLogNEI + prop. MC-EI \\
      Candidates & $4{,}096$ Sobol \\
      Geometry & anisotropic trust region \\
      Trust region & $L_{\mathrm{init}}=0.6$, $L_{\min}=0.05$, $L_{\max}=1.0$ \\
      Success rule & $\tau_{\mathrm{succ}}=3$ \\
      Failure rule & $\alpha=0.5$, $\tau_{\mathrm{fail}}^{\max}=8$ \\
      Shell / exempt & $k_\sigma=2$, $\rho_{\mathrm{exempt}}=0.15$ \\
      Diffuse / sparse & $\rho_{\mathrm{diff}}=0.20$, $20/512$ axes \\
      Batch / DPP & top-$256$, $w_z=0.05$ \\
      Restart / polish & pool or mini-MCTS; $T_{\mathrm{pol}}=35$ \\
      Plateau / cap & $K_{\mathrm{plat}}=5$, $r_{\max}=3$ \\
      \bottomrule
    \end{tabular}
  \end{minipage}
\end{table*}



This subsection records the implementation choices that are intentionally omitted from Sec.~\ref{sec:meridian}. The emphasis here is not on re-deriving the main algorithm, but on documenting the concrete engineering choices that make the method stable in the expensive-oracle regime. All numerical settings appear in Tab.~\ref{tab:hparams-opt} and the committed values in \texttt{config/az31\_*\_meridian\_v2.yaml}. The surrogate stack is a deep-kernel model in which a two-hidden-layer MLP $\phi_\theta : \R^{512} \to \R^{16}$ (widths $[512,128]$, LayerNorm + GELU) is trained jointly with the feasibility classifier $g_\psi$ on \emph{all} observations, preserving the decode/codec/simulator failure signal, and the exact GP is then refit only on the feasible subset. For target-driven $V2$, the same feature trunk also feeds a heteroscedastic linear property head $h_p : \R^{16} \to \R^{2\cdot 5}$ that predicts $(\mu,\log\sigma^2)$ for $(\sigma_y,\sigma_u,n,K,n_{\mathrm{grains}})$ in standardised units. The acquisition used in practice is therefore
\begin{equation}
  \alpha(z) =
  \begin{cases}
    \alpha_{\mathrm{GP}}(z), & V1,\\
    \tfrac{1}{2}\bigl[\tilde\alpha_{\mathrm{GP}}(z) + \tilde\alpha_{\mathrm{prop}}(z)\bigr], & V2,
  \end{cases}
  \qquad
  \alpha_{\mathrm{GP}}(z) = \mathrm{qLogNEI}(z;\mu,\sigma)\, g_\psi(z),
\end{equation}
where $\tilde\alpha_{\mathrm{GP}}$ and $\tilde\alpha_{\mathrm{prop}}$ are $z$-standardised scores and $\tilde\alpha_{\mathrm{prop}}$ is estimated by Monte-Carlo EI with $64$ samples. The property-head linear layer is reset every $K_{\mathrm{reset}}{=}10$ refits to avoid variance collapse on a saturated cache, while the shared feature trunk is warm-started throughout.

Proposal generation is tied explicitly to the latent manifold learned by the decoder. After the first two rounds, search anisotropy is read off from the active-subspace matrix
\begin{equation}
  \hat C = \frac{1}{N}\sum_{i=1}^{N} \nabla \hat\mu(z_i)\, \nabla \hat\mu(z_i)^\top,
\end{equation}
whose leading eigenspace is truncated at $95\%$ cumulative energy and reprojected to ambient axis weights $w\in\R^{512}$; during cold start, that role is played instead by PCA on the class-conditional latent pool $\mathcal P$. Around the shell-projected centroid of the top-$K{=}5$ feasible incumbents, \meridian{} draws a Sobol cloud of $N{=}4096$ candidates, perturbs only a sparse mask of axes with $\Pr[M_{ij}{=}1]=20/d$, and replaces a fraction $\rho_{\mathrm{diff}}{=}0.20$ with isotropic Gaussian moves at scale $0.5L_t\bar w$ so that diffuse high-value pockets are not ruled out a priori. Most proposals are then clipped to the adaptive shell
\begin{equation}
  \|z\| \in [\mu_{\|X\|} \pm k_\sigma \, \sigma_{\|X\|}],
  \qquad k_\sigma = 2,
\end{equation}
estimated from the feasible samples, while a small exempt fraction $\rho_{\mathrm{exempt}}{=}0.15$ is left off-shell in case the support genuinely needs to widen. The trust-region edge evolves in the range $L_t\in[0.05,1.0]$ with $L_{\mathrm{init}}{=}0.6$ under the same success/failure cadence used by \turbo{}/\baxus{}.

Batch construction and restart are likewise tuned for the expensive-oracle regime rather than for formal neatness. From the top-$256$ candidates by acquisition value, a greedy DPP selects the final batch with kernel
\begin{equation}
  L_{ij} = \alpha_i\alpha_j\Bigl[(1-w_z) \, k_{\mathrm{GP}}\bigl(\phi(z_i),\phi(z_j)\bigr) + w_z \, k_{\mathrm{RBF}}(z_i,z_j)\Bigr],
  \qquad w_z = 0.05,
\end{equation}
so diversity is enforced mainly in feature space without giving up direct latent-space separation. For $V2$, one slot is overwritten by a short property-gradient seed and one by a jittered incumbent-polish seed. Past $T_{\mathrm{pol}}{=}35$ of $T{=}40$ rounds, the diffuse fraction, shell exemption, and latent-RBF blend are all set to zero and $L_t$ is capped at $0.1$, yielding a deliberately local polishing phase. A restart is triggered either by trust-region collapse ($L_t<L_{\min}$) or by a plateau of $K_{\mathrm{plat}}{=}5$ rounds; the next anchor is drawn from the class pool or from a small surrogate-side mini-MCTS, the property heads are reset, and termination occurs after $r_{\max}{=}3$ unsuccessful restarts. \refstepcounter{algorithm}\label{alg:meridian-full}\noindent\textbf{Algorithm~\thealgorithm.} This narrative description is the full supplementary specification referenced from the main text.


We use each baseline at the configuration recommended by its authors, with the two pipeline-level adaptations described above (failure masking; restart cap) applied uniformly across all four optimizers, plus the two manifold-awareness adjustments to \baxus{} (PCA-of-seeds embedding, soft-floor reject feedback) detailed in the \baxus{} subsection above. The committed hyperparameter values are listed in Tab.~\ref{tab:hparams-opt}.

\section{Reproducibility checklist mapping}
\label{app:repro}

\begin{table}[H]
  \caption{Mapping of NeurIPS 2026 reproducibility checklist items to the section, file, or asset that satisfies them.}
  \label{tab:repro}
  \centering
  \small
  \begin{tabular}{lll}
    \toprule
    Checklist item & Where addressed & Asset \\
    \midrule
    Code released                  & Sec.~\ref{sec:conclusion}        & supplementary zip \\
    Decoder hyperparameters        & App.~\ref{app:enc-dec}           & Tab.~\ref{tab:hparams-decoder} \\
    Simulator hyperparameters      & App.~\ref{app:damask}            & Tab.~\ref{tab:hparams-sim} \\
    Optimizer hyperparameters      & App.~\ref{app:baselines}         & Tab.~\ref{tab:hparams-opt} \\
    Compute disclosed              & Sec.~\ref{sec:e1}; Sec.~\ref{sec:e2}; App.~\ref{app:extended-results}       & --- \\
    Limitations stated             & Sec.~\ref{sec:conclusion}        & --- \\
    Training Dataset               & Sec.~\ref{sec:e1}                & supplementary zip \\
    \bottomrule
  \end{tabular}
\end{table}

\newpage
\input{checklist.tex}

\end{document}
