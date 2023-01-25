# 3D CNN

## Data Generation
The working directory in `baseline`, all paths mentioned below are in this directory.
### Raw files downloading
Data for this baseline is downloaded from [Protein Data Bank](https://www.rcsb.org).  

Script for downloading PDB data can be found in `download/rsyncPDB.sh`.  

Please make sure to change script so that 
desired version of data is download. Downloaded data should be stored in diretory named 
`pdb_raw_files`, or user can create a directory with a personalized name but make sure 
to change path in `config/data/data.yaml/raw_pdb_dir`.

### Voxel Data Generation
Raw PDB files will be processed by `Biopython`, every residue in PDB files will be converted to a 4-channel binary
 voxel with size `(4, 20, 20, 20)`. Carbon, Nitrogen, Oxygen and Sulfur represent one of the four channels, 
respectively. All voxels have the same orientation, boxes are translated to put carbon beta at the centered and rotated
so that vector parallelized to Carbon_beta -- Carbon_alpha is `Vector(1,1,1)`.

### HDF5 Data Generation
All voxel boxes are stored in the format of hdf5 files. Voxel boxes coming from the same PDB file are stored in the same
hdf5 file. All hdf5 files can be found in directory `voxel_hdf5`.The layout of this directory is:   
+-- voxel_hdf5  
|   +-- PDB0.hdf5   
|   +-- PDB1.hdf5   
...     
...         
|   +-- PDBx.hdf5   

The layout inside one specific PDBx.hdf5 is:    
|+-- /   
|   +-- group_Chain_ID = "A"      
|      +-- dataset_residue_serial_number = "1"         
|      +-- dataset_residue_serial_number = "2"         
...         
...             
|   +-- group_Chain_ID = "B"      
|      +-- dataset_residue_serial_number = "1"         
|          +-- dataset_residue_serial_number = "2"      
...     


To generate all hdf5 file, run the command line         
`cd voxel_box`      
`python data_gen.py`





