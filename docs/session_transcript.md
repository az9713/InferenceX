# Terminal Session: Generating 25 Benchmark Jobs from InferenceX

This is the actual terminal session output from running the InferenceX config
generator locally on a laptop (no GPU required). It shows:
1. Installing dependencies
2. Running 115 tests (all pass)
3. Generating 25 benchmark job configurations for DeepSeek-R1 on NVIDIA B200

See [HANDS_ON_WALKTHROUGH.md](HANDS_ON_WALKTHROUGH.md) for a detailed
explanation of every command, flag, and output field.

---

## Session Output

```
$ cd InferenceX
$ source venv/Scripts/activate
(venv)
$ pip install pydantic pyyaml pytest
Collecting pydantic
  Using cached pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
Collecting pyyaml
  Using cached pyyaml-6.0.3-cp313-cp313-win_amd64.whl.metadata (2.4 kB)
Collecting pytest
  Using cached pytest-9.0.2-py3-none-any.whl.metadata (7.6 kB)
Collecting annotated-types>=0.6.0 (from pydantic)
  Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
Collecting pydantic-core==2.41.5 (from pydantic)
  Using cached pydantic_core-2.41.5-cp313-cp313-win_amd64.whl.metadata (7.4 kB)
Collecting typing-extensions>=4.14.1 (from pydantic)
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting typing-inspection>=0.4.2 (from pydantic)
  Using cached typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
Collecting colorama>=0.4 (from pytest)
  Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
Collecting iniconfig>=1.0.1 (from pytest)
  Using cached iniconfig-2.3.0-py3-none-any.whl.metadata (2.5 kB)
Collecting packaging>=22 (from pytest)
  Using cached packaging-26.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pluggy<2,>=1.5 (from pytest)
  Using cached pluggy-1.6.0-py3-none-any.whl.metadata (4.8 kB)
Collecting pygments>=2.7.2 (from pytest)
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Using cached pydantic-2.12.5-py3-none-any.whl (463 kB)
Using cached pydantic_core-2.41.5-cp313-cp313-win_amd64.whl (2.0 MB)
Using cached pyyaml-6.0.3-cp313-cp313-win_amd64.whl (154 kB)
Using cached pytest-9.0.2-py3-none-any.whl (374 kB)
Using cached pluggy-1.6.0-py3-none-any.whl (20 kB)
Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
Using cached iniconfig-2.3.0-py3-none-any.whl (7.5 kB)
Using cached packaging-26.0-py3-none-any.whl (74 kB)
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Using cached typing_inspection-0.4.2-py3-none-any.whl (14 kB)
Installing collected packages: typing-extensions, pyyaml, pygments, pluggy, packaging,
  iniconfig, colorama, annotated-types, typing-inspection, pytest, pydantic-core, pydantic
Successfully installed annotated-types-0.7.0 colorama-0.4.6 iniconfig-2.3.0
  packaging-26.0 pluggy-1.6.0 pydantic-2.12.5 pydantic-core-2.41.5
  pygments-2.19.2 pytest-9.0.2 pyyaml-6.0.3 typing-extensions-4.15.0
  typing-inspection-0.4.2
(venv)
$ cd utils
(venv)
$ python -m pytest matrix_logic/ -v
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.0.2, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: <project-root>/utils/matrix_logic
configfile: pytest.ini
collected 115 items

matrix_logic/test_generate_sweep_configs.py::TestSeqLenMappings::test_seq_len_stoi_values PASSED [  0%]
matrix_logic/test_generate_sweep_configs.py::TestSeqLenMappings::test_seq_len_itos_reverse_mapping PASSED [  1%]
matrix_logic/test_generate_sweep_configs.py::TestSeqLenToStr::test_known_sequence_lengths PASSED [  2%]
matrix_logic/test_generate_sweep_configs.py::TestSeqLenToStr::test_unknown_sequence_lengths PASSED [  3%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_basic_sweep_generation PASSED [  4%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_matrix_entry_structure PASSED [  5%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_filter_by_model_prefix PASSED [  6%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_filter_by_precision PASSED [  6%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_filter_by_framework PASSED [  7%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_filter_by_runner_type PASSED [  8%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_invalid_runner_type_raises_error PASSED [  9%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_filter_by_seq_lens PASSED [ 10%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_max_conc_filter PASSED [ 11%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_max_conc_creates_config_when_below_min PASSED [ 12%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_max_conc_zero_or_negative_skips PASSED [ 13%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_max_tp_filter PASSED [ 13%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_max_tp_below_all_available_skips PASSED [ 14%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_max_tp_zero_or_negative_skips PASSED [ 15%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_step_size PASSED [ 16%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_exp_name_format PASSED [ 17%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_max_model_len_calculation PASSED [ 18%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_runner_node_filter PASSED [ 19%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_runner_node_filter_no_match PASSED [ 20%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_runner_node_filter_without_runner_type PASSED [ 20%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepMultiNode::test_multinode_sweep_generation PASSED [ 21%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepMultiNode::test_multinode_entry_structure PASSED [ 22%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepMultiNode::test_multinode_conc_as_list PASSED [ 23%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepMultiNode::test_single_node_flag_skips_multinode PASSED [ 24%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepMultiNode::test_runner_node_filter_multinode PASSED [ 25%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_basic_runner_sweep PASSED [ 26%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_runner_sweep_entry_structure PASSED [ 26%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_each_node_gets_entry PASSED [ 27%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_invalid_runner_type PASSED [ 28%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_runner_node_filter PASSED [ 29%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_runner_node_filter_no_match PASSED [ 30%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_uses_highest_tp PASSED [ 31%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_uses_lowest_conc PASSED [ 32%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_filter_by_model_prefix PASSED [ 33%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_filter_by_precision PASSED [ 33%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_filter_by_framework PASSED [ 34%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_combined_filters PASSED [ 35%]
matrix_logic/test_generate_sweep_configs.py::TestGenerateRunnerModelSweepConfig::test_conc_override PASSED [ 36%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_config_with_ep_and_dp_attn PASSED [ 37%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_config_with_spec_decoding PASSED [ 38%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_conc_list_in_single_node PASSED [ 39%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_disagg_defaults_to_false PASSED [ 40%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_multinode_conc_range_expansion PASSED [ 40%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_max_ep_creates_config_when_below_min PASSED [ 41%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_max_ep_zero_or_negative_skips PASSED [ 42%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_multinode_max_conc_zero_or_negative_skips PASSED [ 43%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_multinode_max_conc_creates_config_when_below_min PASSED [ 44%]
matrix_logic/test_generate_sweep_configs.py::TestEdgeCases::test_combined_max_filters PASSED [ 45%]
matrix_logic/test_generate_sweep_configs.py::TestArgumentDefaults::test_runner_config_default_value PASSED [ 46%]
matrix_logic/test_generate_sweep_configs.py::TestArgumentDefaults::test_runner_config_explicit_value PASSED [ 46%]
matrix_logic/test_validation.py::TestFieldsEnum::test_field_values_are_strings PASSED [ 47%]
matrix_logic/test_validation.py::TestFieldsEnum::test_key_fields_exist PASSED [ 48%]
matrix_logic/test_validation.py::TestWorkerConfig::test_valid_worker_config PASSED [ 49%]
matrix_logic/test_validation.py::TestWorkerConfig::test_worker_config_with_additional_settings PASSED [ 50%]
matrix_logic/test_validation.py::TestWorkerConfig::test_worker_config_missing_required_field PASSED [ 51%]
matrix_logic/test_validation.py::TestWorkerConfig::test_worker_config_extra_field_forbidden PASSED [ 52%]
matrix_logic/test_validation.py::TestSingleNodeMatrixEntry::test_valid_entry PASSED [ 53%]
matrix_logic/test_validation.py::TestSingleNodeMatrixEntry::test_conc_as_list PASSED [ 53%]
matrix_logic/test_validation.py::TestSingleNodeMatrixEntry::test_spec_decoding_values PASSED [ 54%]
matrix_logic/test_validation.py::TestSingleNodeMatrixEntry::test_invalid_spec_decoding PASSED [ 55%]
matrix_logic/test_validation.py::TestSingleNodeMatrixEntry::test_missing_required_field PASSED [ 56%]
matrix_logic/test_validation.py::TestSingleNodeMatrixEntry::test_extra_field_forbidden PASSED [ 57%]
matrix_logic/test_validation.py::TestMultiNodeMatrixEntry::test_valid_entry PASSED [ 58%]
matrix_logic/test_validation.py::TestMultiNodeMatrixEntry::test_prefill_decode_worker_configs PASSED [ 59%]
matrix_logic/test_validation.py::TestMultiNodeMatrixEntry::test_conc_must_be_list PASSED [ 60%]
matrix_logic/test_validation.py::TestMultiNodeMatrixEntry::test_missing_prefill PASSED [ 60%]
matrix_logic/test_validation.py::TestMultiNodeMatrixEntry::test_missing_decode PASSED [ 61%]
matrix_logic/test_validation.py::TestValidateMatrixEntry::test_valid_single_node PASSED [ 62%]
matrix_logic/test_validation.py::TestValidateMatrixEntry::test_valid_multinode PASSED [ 63%]
matrix_logic/test_validation.py::TestValidateMatrixEntry::test_invalid_single_node_raises_valueerror PASSED [ 64%]
matrix_logic/test_validation.py::TestValidateMatrixEntry::test_invalid_multinode_raises_valueerror PASSED [ 65%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_valid_with_conc_range PASSED [ 66%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_valid_with_conc_list PASSED [ 66%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_cannot_have_both_range_and_list PASSED [ 67%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_must_have_range_or_list PASSED [ 68%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_conc_start_must_be_lte_conc_end PASSED [ 69%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_conc_list_values_must_be_positive PASSED [ 70%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_optional_fields_defaults PASSED [ 71%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_with_ep_and_dp_attn PASSED [ 72%]
matrix_logic/test_validation.py::TestSingleNodeSearchSpaceEntry::test_with_spec_decoding_mtp PASSED [ 73%]
matrix_logic/test_validation.py::TestMultiNodeSearchSpaceEntry::test_valid_with_conc_list PASSED [ 73%]
matrix_logic/test_validation.py::TestMultiNodeSearchSpaceEntry::test_valid_with_conc_range PASSED [ 74%]
matrix_logic/test_validation.py::TestMultiNodeSearchSpaceEntry::test_with_spec_decoding_mtp PASSED [ 75%]
matrix_logic/test_validation.py::TestMultiNodeSearchSpaceEntry::test_missing_conc_specification PASSED [ 76%]
matrix_logic/test_validation.py::TestSeqLenConfigs::test_single_node_seq_len_config_1k1k PASSED [ 77%]
matrix_logic/test_validation.py::TestSeqLenConfigs::test_single_node_seq_len_config_8k1k PASSED [ 78%]
matrix_logic/test_validation.py::TestSeqLenConfigs::test_multinode_seq_len_config PASSED [ 79%]
matrix_logic/test_validation.py::TestMasterConfigEntries::test_single_node_master_config PASSED [ 80%]
matrix_logic/test_validation.py::TestMasterConfigEntries::test_multinode_master_config PASSED [ 80%]
matrix_logic/test_validation.py::TestMasterConfigEntries::test_single_node_cannot_have_multinode_true PASSED [ 81%]
matrix_logic/test_validation.py::TestMasterConfigEntries::test_multinode_cannot_have_multinode_false PASSED [ 82%]
matrix_logic/test_validation.py::TestMasterConfigEntries::test_disagg_default_false PASSED [ 83%]
matrix_logic/test_validation.py::TestValidateMasterConfig::test_valid_single_node_config PASSED [ 84%]
matrix_logic/test_validation.py::TestValidateMasterConfig::test_valid_multinode_config PASSED [ 85%]
matrix_logic/test_validation.py::TestValidateMasterConfig::test_mixed_configs PASSED [ 86%]
matrix_logic/test_validation.py::TestValidateMasterConfig::test_invalid_config_raises_valueerror PASSED [ 86%]
matrix_logic/test_validation.py::TestValidateRunnerConfig::test_valid_runner_config PASSED [ 87%]
matrix_logic/test_validation.py::TestValidateRunnerConfig::test_value_must_be_list PASSED [ 88%]
matrix_logic/test_validation.py::TestValidateRunnerConfig::test_list_must_contain_strings PASSED [ 89%]
matrix_logic/test_validation.py::TestValidateRunnerConfig::test_list_cannot_be_empty PASSED [ 90%]
matrix_logic/test_validation.py::TestValidateRunnerConfig::test_multiple_runner_types PASSED [ 91%]
matrix_logic/test_validation.py::TestLoadConfigFiles::test_load_single_file_with_validation PASSED [ 92%]
matrix_logic/test_validation.py::TestLoadConfigFiles::test_load_single_file_without_validation PASSED [ 93%]
matrix_logic/test_validation.py::TestLoadConfigFiles::test_load_multiple_files PASSED [ 93%]
matrix_logic/test_validation.py::TestLoadConfigFiles::test_duplicate_keys_raise_error PASSED [ 94%]
matrix_logic/test_validation.py::TestLoadConfigFiles::test_nonexistent_file_raises_error PASSED [ 95%]
matrix_logic/test_validation.py::TestLoadConfigFiles::test_validation_runs_by_default PASSED [ 96%]
matrix_logic/test_validation.py::TestLoadRunnerFile::test_load_runner_file_with_validation PASSED [ 97%]
matrix_logic/test_validation.py::TestLoadRunnerFile::test_load_runner_file_without_validation PASSED [ 98%]
matrix_logic/test_validation.py::TestLoadRunnerFile::test_nonexistent_runner_file PASSED [ 99%]
matrix_logic/test_validation.py::TestLoadRunnerFile::test_validation_runs_by_default PASSED [100%]

============================= 115 passed in 1.29s =============================
(venv)
$ cd ..
(venv)
$ python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/nvidia-master.yaml \
    --single-node \
    --model-prefix dsr1 \
    --runner-type b200 \
    --seq-lens 1k1k \
    | python -m json.tool
[
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 4,
        "conc": 4,
        "max-model-len": 2248,
        "ep": 4,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 4,
        "conc": 8,
        "max-model-len": 2248,
        "ep": 4,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 4,
        "conc": 16,
        "max-model-len": 2248,
        "ep": 4,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 4,
        "conc": 32,
        "max-model-len": 2248,
        "ep": 4,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 4,
        "conc": 64,
        "max-model-len": 2248,
        "ep": 4,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 4,
        "conc": 128,
        "max-model-len": 2248,
        "ep": 4,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 4,
        "max-model-len": 2248,
        "ep": 8,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 8,
        "max-model-len": 2248,
        "ep": 8,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 16,
        "max-model-len": 2248,
        "ep": 8,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 32,
        "max-model-len": 2248,
        "ep": 8,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 64,
        "max-model-len": 2248,
        "ep": 8,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
        "model-prefix": "dsr1",
        "precision": "fp4",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 128,
        "max-model-len": 2248,
        "ep": 8,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 4,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 8,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 16,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 32,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 64,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "none",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 4,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 8,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 16,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 32,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 64,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 128,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 256,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    },
    {
        "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "model-prefix": "dsr1",
        "precision": "fp8",
        "framework": "sglang",
        "runner": "b200",
        "isl": 1024,
        "osl": 1024,
        "tp": 8,
        "conc": 512,
        "max-model-len": 2248,
        "ep": 1,
        "dp-attn": false,
        "spec-decoding": "mtp",
        "exp-name": "dsr1_1k1k",
        "disagg": false,
        "run-eval": false
    }
]
```
