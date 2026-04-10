"""Write Snakemake config.yaml from pipeline defaults."""
from __future__ import annotations

import os

import yaml

from .paths import DATA_INPUTS, GEX_PATH, PIPELINE_ROOT, SNAKE_DIR


def write_snakemake_config(args):
    """Write config.yaml under Snakemake dir so pipeline paths are the single source of truth."""
    n_cpu = getattr(args, "cores", 4)
    config = {
        "input_data": {
            "cisTopic_obj_fname": os.path.join(DATA_INPUTS, "cistopic_obj.pkl"),
            "GEX_anndata_fname": GEX_PATH,
            "region_set_folder": os.path.join(DATA_INPUTS, "region_sets"),
            "ctx_db_fname": os.path.join(DATA_INPUTS, "hg38_screen_v10_clust.regions_vs_motifs.rankings.feather"),
            "dem_db_fname": os.path.join(DATA_INPUTS, "hg38_screen_v10_clust.regions_vs_motifs.scores.feather"),
            "path_to_motif_annotations": os.path.join(DATA_INPUTS, "motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl.txt"),
        },
        "output_data": {
            "combined_GEX_ACC_mudata": "ACC_GEX.h5mu",
            "dem_result_fname": "dem_results.hdf5",
            "ctx_result_fname": "ctx_results.hdf5",
            "output_fname_dem_html": "dem_results.html",
            "output_fname_ctx_html": "ctx_results.html",
            "cistromes_direct": "cistromes_direct.h5ad",
            "cistromes_extended": "cistromes_extended.h5ad",
            "tf_names": "tf_names.txt",
            "genome_annotation": "genome_annotation.tsv",
            "chromsizes": os.path.join(SNAKE_DIR, "chromsizes.tsv"),
            "search_space": "search_space.tsv",
            "tf_to_gene_adjacencies": "tf_to_gene_adj.tsv",
            "region_to_gene_adjacencies": "region_to_gene_adj.tsv",
            "eRegulons_direct": "eRegulon_direct.tsv",
            "eRegulons_extended": "eRegulons_extended.tsv",
            "AUCell_direct": "AUCell_direct.h5mu",
            "AUCell_extended": "AUCell_extended.h5mu",
            "scplus_mdata": "scplusmdata.h5mu",
        },
        "params_general": {
            "temp_dir": os.path.join(PIPELINE_ROOT, "tmp"),
            "n_cpu": n_cpu,
            "seed": 666,
        },
        "params_data_preparation": {
            "bc_transform_func": '"lambda x: f\'{x}___cisTopic\'"',
            "is_multiome": True,
            "key_to_group_by": "",
            "nr_cells_per_metacells": 10,
            "direct_annotation": "Direct_annot",
            "extended_annotation": "Orthology_annot",
            "species": "hsapiens",
            "biomart_host": "http://www.ensembl.org",
            "search_space_upstream": "1000 150000",
            "search_space_downstream": "1000 150000",
            "search_space_extend_tss": "10 10",
        },
        "params_motif_enrichment": {
            "species": "homo_sapiens",
            "annotation_version": "v10nr_clust",
            "motif_similarity_fdr": 0.001,
            "orthologous_identity_threshold": 0.0,
            "annotations_to_use": "Direct_annot Orthology_annot",
            "fraction_overlap_w_dem_database": 0.2,
            "dem_max_bg_regions": 5000,
            "dem_balance_number_of_promoters": True,
            "dem_promoter_space": "1_000",
            "dem_adj_pval_thr": 0.1,
            "dem_log2fc_thr": 0.5,
            "dem_mean_fg_thr": 0.0,
            "dem_motif_hit_thr": 1.0,
            "fraction_overlap_w_ctx_database": 0.4,
            "ctx_auc_threshold": 0.005,
            "ctx_nes_threshold": 3.0,
            "ctx_rank_threshold": 0.05,
        },
        "params_inference": {
            "tf_to_gene_importance_method": "GBM",
            "region_to_gene_importance_method": "GBM",
            "region_to_gene_correlation_method": "SR",
            "order_regions_to_genes_by": "importance",
            "order_TFs_to_genes_by": "importance",
            "gsea_n_perm": 1000,
            "quantile_thresholds_region_to_gene": "0.85 0.90 0.95",
            "top_n_regionTogenes_per_gene": "5 10 15",
            "top_n_regionTogenes_per_region": "",
            "min_regions_per_gene": 1,
            "rho_threshold": 0.05,
            "min_target_genes": 10,
        },
    }
    config_dir = os.path.join(SNAKE_DIR, "config")
    os.makedirs(config_dir, exist_ok=True)
    path = os.path.join(config_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
