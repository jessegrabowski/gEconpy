% Regenerate tests/_resources/dynare_outputs from the .mod files in this directory.
%
% Run from any directory, with Dynare's matlab folder on the path:
%   octave --no-gui --eval "addpath('<dynare>/matlab'); run('tests/_resources/dynare_models/regenerate_fixtures.m')"
%
% Dynare writes its generated files next to the .mod file, so each model runs in a temporary copy.
%
% Each second-order model also gets a pruned simulation along a fixed shock path, written to <model>_pruned_path.mat.
% The shocks are a deterministic function of the period so that the test reads the same path back.

fixture_models = {'basic_rbc', 'basic_rbc_loglinear', 'basic_rbc_logvars', 'full_nk', 'one_block_1_ss', ...
                  'one_block_1_ss_2shock', 'rbc_2_block_ss'};

model_dir = fileparts(mfilename('fullpath'));
output_dir = fullfile(model_dir, '..', 'dynare_outputs');
work_dir = tempname();
mkdir(work_dir);
start_dir = pwd();
cd(work_dir);

for i = 1:numel(fixture_models)
    model_name = fixture_models{i};
    copyfile(fullfile(model_dir, [model_name '.mod']), work_dir);
    dynare(model_name, 'noclearall', 'nolog');
    save('-v7', fullfile(output_dir, [model_name '_results.mat']), 'oo_', 'M_');

    if options_.order == 2
        pruned_shocks = 3 * sin((1:40)' * (1:M_.exo_nbr)) .* sqrt(diag(M_.Sigma_e))';
        pruned_path = simult_(M_, options_, oo_.dr.ys, oo_.dr, pruned_shocks, 2);
        save('-v7', fullfile(output_dir, [model_name '_pruned_path.mat']), 'pruned_shocks', 'pruned_path');
    end
end

cd(start_dir);
rmdir(work_dir, 's');
