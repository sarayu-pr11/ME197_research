import h5py

file_path = "data/emg2pose_dataset_mini/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_left.hdf5"
with h5py.File(file_path, "r") as f:
    dset = f["emg2pose/timeseries"]
    print("Dataset shape:", dset.shape)
    print("Dataset dtype:", dset.dtype)
    print("First row sample:", dset[0])
