# Repository for: Computational immunohistochemical and H&E mapping adds immune context to histological phenotypes in colitis mouse models.

## [Soma Kobayashi, Christopher Sullivan, , Agnieszka B. Bialkowska, Joel H. Saltz, Vincent W. Yang]

### Environment Setup: Please refer to INSTRUCTIONS.md in ENVIRONMENT folder

### Please download ROI_TIFs to run code at: https://drive.google.com/drive/folders/1m9O5uOvz2-aAelVJy0xdF2w-rJr_H8p5?usp=drive_link
--> ROI_tifs folder should be saved and absolute path to it should be used for data['directories']['ROI_INPUT_DIR'] in the generateJSON.py folder

### Please download the following two CSVs: 1) involved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv and 2) UNinvolved_patch_RN_FeatureExtraction/UNinvolved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv at: https://drive.google.com/drive/u/0/folders/1O6Bk57bmPSBSxkwlMpTY9pLQ9btt0laJ
--> please place both in ./6_mouseModelInference/RN_featureextraction_features/

### WSIs and involved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv download available at: https://drive.google.com/drive/folders/1O6Bk57bmPSBSxkwlMpTY9pLQ9btt0laJ?usp=drive_link
### Please move the involved_wOverlap_RN_extractedFeatures_archivedMouseCohort.csv file to './6_mouseModelInference/archivedMouseCohort_InvolvedPatches_RN_extracted_features'

### Running code:
### - Edit input and base dir paths in generateJSON.py
### - python run_pipeline.py 

