% Convert GSH coefficient files -> DREAM.3D uniform ODF angle files.
% Serial, env-driven version of the validated Sampling script
% (MTEX/SHcoeffs_to_odf/SHcoeffs_to_Dream3d_odf_batch_parallel.m):
%   sigma    = 1.0    (sigma < 1 breaks MatchCrystallography; > 1 over-smooths)
%   nSamples = 100000 (raw discreteSample orientations, uniform weights)
%
% Environment variables:
%   MTEX_ROOT      - MTEX toolbox root (e.g. ~/Documents/MTEX/mtex-5.11.2)
%   ODF_PARENT_DIR - scanned recursively for *.txt harmonics files; outputs go
%                    to <dir>/dream3d_angles_uniform/<base>_dream3d_odf_angles.txt
%
% Run: matlab -batch "run('/abs/path/to/harmonics_to_dream3d_angles.m')"

mtexRoot  = getenv('MTEX_ROOT');
parentDir = getenv('ODF_PARENT_DIR');
assert(~isempty(mtexRoot) && ~isempty(parentDir), ...
    'Set MTEX_ROOT and ODF_PARENT_DIR environment variables');

addpath(mtexRoot);
startup_mtex;

setMTEXpref('EulerAngleConvention', 'Bunge');
setMTEXpref('xAxisDirection', 'east');
setMTEXpref('zAxisDirection', 'outOfPlane');
setMTEXpref('quiet', true);
setMTEXpref('useNFFT', false);

nSamples  = 100000;
sigmaVal  = 1.0;
rngSeed   = 1;
outSubDir = 'dream3d_angles_uniform';

CS = crystalSymmetry('6/mmm', [3.2093 3.2093 5.2103], ...
    'X||a*', 'Y||b', 'Z||c', 'mineral', 'Mg');
SS = specimenSymmetry('1');

files = dir(fullfile(parentDir, '**', '*.txt'));
nDone = 0;
nSkip = 0;
for i = 1:numel(files)
    inName = files(i).name;
    inDir  = files(i).folder;

    if contains(inName, '_ODF.txt') || contains(inName, '_dream3d_odf_angles.txt')
        continue;
    end

    [~, baseName, ~] = fileparts(inName);
    outDir  = fullfile(inDir, outSubDir);
    outFile = fullfile(outDir, [baseName '_dream3d_odf_angles.txt']);

    if isfile(outFile)
        nSkip = nSkip + 1;
        continue;
    end
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    inFile = fullfile(inDir, inName);
    rng(rngSeed + i, 'twister');

    try
        data = readmatrix(inFile);
        data(any(isnan(data), 2), :) = [];
        if isempty(data) || size(data, 2) < 2
            error('Input file is empty or missing real/imag columns: %s', inFile);
        end
        fhat = complex(data(:,1), data(:,2));
        fhat = fhat(:);

        try
            odf_recons = SO3FunHarmonic(fhat, CS, SS);
        catch
            base_odf   = SO3FunHarmonic(fhat);
            odf_recons = SO3FunHarmonic(base_odf.fhat, CS, SS);
        end

        ori = discreteSample(odf_recons, nSamples);
        [phi1, Phi, phi2] = Euler(ori, 'Bunge');
        phi1 = mod(phi1 ./ degree, 360.0);
        Phi  = mod(Phi  ./ degree, 180.0);
        phi2 = mod(phi2 ./ degree, 360.0);

        weights = ones(nSamples, 1);
        sigmas  = sigmaVal * ones(nSamples, 1);
        outData = [phi1(:), Phi(:), phi2(:), weights, sigmas];

        fid = fopen(outFile, 'w');
        if fid < 0
            error('Cannot open output file: %s', outFile);
        end
        fprintf(fid, '# DREAM.3D StatsGenerator Angles Input File\n');
        fprintf(fid, '# Generated from GSH coefficients via MTEX ODF reconstruction\n');
        fprintf(fid, '# Export mode: raw discreteSample (no binning)\n');
        fprintf(fid, '# Source: %s\n', inFile);
        fprintf(fid, '# Euler0 Euler1 Euler2 Weight Sigma\n');
        fprintf(fid, 'Angle Count:%d\n', nSamples);
        fprintf(fid, '%.6f %.6f %.6f %.6f %.6f\n', outData.');
        fclose(fid);

        nDone = nDone + 1;
        fprintf('OK  %s\n', outFile);
    catch ME
        fprintf(2, 'FAIL %s: %s\n', inFile, ME.message);
    end
end
fprintf('Done. converted=%d skipped_existing=%d\n', nDone, nSkip);
