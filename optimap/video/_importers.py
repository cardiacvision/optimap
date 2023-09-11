import locale
import struct
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np

class MultiRecorderImporter:
    """Importer for MultiRecorder (MPI-DS) recordings (.dat files)"""
    _Nx, _Ny, _Nt = 0, 0, 0
    _header_size = 1024
    _skip_per_frame = 0
    _endian = "<"
    _meta = {}


    def __init__(self, filepath, is_8bit=False):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File {self.filepath} not found")
        
        self.is_8bit = is_8bit
        with open(self.filepath, "rb") as f:
            self._read_header(f)

    def get_metadata(self):
        """Returns the metadata dictionary"""
        return self._meta
    
    def load_video(self, start_frame=0, frames=None, step=1, use_mmap=False):
        """
        Returns a 3D numpy array containing the loaded video.
        """

        if frames is not None:
            nframes = frames
            if nframes > self._Nt - start_frame:
                print(f"Warning: requested {nframes} frames, but only {self._Nt - start_frame} frames available. Loading all available frames.")
                nframes = self._Nt - start_frame
        else:
            nframes = self._Nt - start_frame
        
        if self.is_8bit:
            dt = np.dtype(f"{self._endian}B")
            bs = 1
        else:
            dt = np.dtype(f"{self._endian}H")
            bs = 2

        bytes_per_frame = self._Nx * self._Ny * bs + self._skip_per_frame
        start_offset = self._header_size + bytes_per_frame * start_frame
        arr = np.memmap(
            self.filepath,
            np.byte,
            mode="r",
            offset=start_offset,
            shape=(bytes_per_frame * nframes, 1),
        )

        # convert to dtype and stride end of frame data
        arr = np.ndarray(
            shape=(nframes, self._Nx, self._Ny),
            dtype=dt,
            buffer=arr,
            offset=0,
            strides=(
                (bs * self._Nx * self._Ny + self._skip_per_frame),
                self._Ny * bs,
                bs,
            ),
        )
        arr = arr[::step]

        if not use_mmap:
            arr = arr.copy("C")

        return arr

    def _read_header(self, f: BinaryIO):
        """
        Read the header
        """
        self.version = f.read(1).decode("utf-8")
        sum = 1
        if self.version == "d":
            date = f.read(17)
            if date[-3:-2] != ":":
                date += f.read(7)
                sum += 7
            self._meta["date"] = date

            self._Nt = struct.unpack(f"{self._endian}i", f.read(4))[0]
            self._Ny = struct.unpack(f"{self._endian}i", f.read(4))[0]
            self._Nx = struct.unpack(f"{self._endian}i", f.read(4))[0]

        elif self.version == "e" or self.version == "f":
            self._skip_per_frame = 8

            en = struct.unpack(">i", f.read(4))[0]
            if en == 439041101:  # 0x1A2B3C4D
                self._endian = ">"
            else:
                self._endian = "<"
            self._Nt = struct.unpack(f"{self._endian}i", f.read(4))[0]
            self._Ny = struct.unpack(f"{self._endian}i", f.read(4))[0]
            self._Nx = struct.unpack(f"{self._endian}i", f.read(4))[0]
            f.read(8)
            self._meta["framerate"] = (
                struct.unpack(f"{self._endian}i", f.read(4))[0] / 100000
            )

            dtime = f.read(24).decode("utf-8").rstrip("\x00")

            # Only works with US local:
            cl = locale.getlocale(locale.LC_TIME)
            locale.setlocale(locale.LC_TIME, (None, None))
            try:
                self._meta["starttime"] = datetime.strptime(dtime, "%c")
            except ValueError:
                try:
                    dtime2 = ":".join(dtime.split(":")[:-1])
                    self._meta["starttime"] = datetime.strptime(
                        dtime2, "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    self._meta["starttime"] = datetime.fromtimestamp(0)
                    print(
                        "Could not determine video starttime from string: " + str(dtime)
                    )
            locale.setlocale(locale.LC_TIME, cl)

            self._meta["comment"] = f.read(971).rstrip(b"\x00").decode("utf-8")
        else:
            raise ValueError(f"Unknown MultiRecorder file version: {self.version}")


class MiCAM05_Importer:
    """
    Importer for SciMedia MiCAM05 CMOS camera recordings (.gsh / .gsd files)

    .. warning:: Tested only on sample MiCAM05-N256 camera files with 256x256 pixels resolution.
    """
    _Nx, _Ny, _Nt = 0, 0, 0
    _dtype = np.uint16
    _gsd_header_size = 970
    _meta = {}

    def __init__(self, filepath):
        filepath = Path(filepath)
        self.gsd = filepath.with_suffix(".gsd")
        self.gsh = filepath.with_suffix(".gsh")
        if not self.gsd.exists():
            raise FileNotFoundError(f"File {self.gsd} not found")
        if not self.gsh.exists():
            raise FileNotFoundError(f"File {self.gsh} not found")
        self._read_header()

    def _read_header(self):
        with open(self.gsh, "rt") as fidHeader:
            header = fidHeader.read()
        header = header.split("\n")
        header = [line.split(":", 1) for line in header]
        self._meta = {}
        for line in header:
            if len(line) == 2 and line[1].strip() != "":
                self._meta[line[0].strip()] = line[1].strip()

        self._Nt = int(self._meta["Number of frames"])
        self._Nx = int(self._meta["Image width"])
        self._Ny = int(self._meta["Image height"])
        self._meta['framerate'] = float(self._meta["Frame rate (Hz)"])
        try:
            self._meta['date'] = datetime.strptime(self._meta["Date created"], '%d/%m/%Y %H:%M:%S')
        except ValueError:
            self._meta['date'] = self._meta["Date created"]

    def load_video(self, start_frame=0, frames=None, step=1):
        if frames is not None:
            end_frame = start_frame + frames
            if end_frame > self._Nt:
                print(f"Warning: requested {frames} frames, but only {self._Nt - start_frame} frames available. Loading all available frames.")
                end_frame = None
        else:
            end_frame = None
        
        background_image = np.fromfile(
                self.gsd,
                dtype=self._dtype,
                count=(self._Nx * self._Ny),
                offset=self._gsd_header_size
            )
        background_image = background_image.reshape((self._Nx, self._Ny))

        CMOS_data = np.memmap(self.gsd,
                              dtype=self._dtype,
                              offset=self._gsd_header_size + background_image.nbytes,
                              shape=(self._Nt, self._Nx, self._Ny),
                              mode='r')
        CMOS_data = CMOS_data[start_frame:end_frame:step].copy()
        video = CMOS_data + background_image[np.newaxis, :, :]
        return video
    
    def get_metadata(self):
        """Returns the metadata dictionary"""
        return self._meta

class MiCAM_ULTIMA_Importer:
    """
    Importer for SciMedia MiCAM ULTIMA recordings (.rsh / .rsm / .rsd files)

    .. warning:: Tested only on a sample MiCAM ULTIMA camera recording. Might not work correctly.
    """
    _Nx, _Ny, _Nt = 0, 0, 0
    _lskp = 0  # left skip
    _rskp = 0  # right skip
    _blk = 0  # frames per block
    _dtype = np.uint16
    _meta = {}

    def __init__(self, filepath):
        filepath = Path(filepath)
        self.rsh = filepath.with_suffix(".rsh")
        self.rsm = filepath.with_suffix(".rsm")
        self.rsd_list = []
        if not self.rsh.exists():
            raise FileNotFoundError(f"File {self.rsh} not found")
        self._read_header()

    def _read_header(self):
        with open(self.rsh, "rt") as fidHeader:
            header = fidHeader.read()
        header = header.split("\n")

        self._parse_topheader(header[1:3])
        idx = header.index('Data-File-List')
        self._parse_metadata(header[3:idx])
        self._parse_filelist(header[idx+1:])
    
    def _parse_topheader(self, header):
        data = {}
        for s in ''.join(header).split('/'):
            if '=' in s:
                key, value = s.split('=', 1)
                data[key] = value
        self._Ny = int(data['x'])
        self._Nx = int(data['y'])
        self._lskp = int(data['lskp'])
        self._rskp = -int(data['rskp'])
        if self._rskp == 0:
            self._rskp = None
        self._blk = int(data['blk'])

    def _parse_metadata(self, header):
        for line in header:
            if '=' in line:
                key, value = line.split('=', 1)
                self._meta[key] = value
        if 'page_frames' in self._meta:
            self._Nt = int(self._meta['page_frames'])
        if 'acquisition_date' in self._meta:
            try:
                self._meta['date'] = datetime.strptime(self._meta['acquisition_date'], '%Y/%m/%d %H:%M:%S')
            except ValueError:
                self._meta['date'] = self._meta['acquisition_date']

    def _parse_filelist(self, header):
        base = self.rsh.parent
        for line in header:
            file = base / line.strip()
            if file.suffix == '.rsm':
                self.rsm = file
            elif file.suffix == '.rsd':
                self.rsd_list.append(file)
    
    def load_video(self, start_frame=0, frames=None, step=1):
        """
        Returns a 3D numpy array containing the loaded video.
        """
        if frames is not None:
            end_frame = start_frame + frames
            if end_frame > self._Nt:
                print(f"Warning: requested {frames} frames, but only {self._Nt - start_frame} frames available. Loading all available frames.")
                end_frame = None
        else:
            end_frame = None

        background = np.fromfile(self.rsm, dtype="<u2")
        background = background.reshape((self._Nx, self._Ny))[np.newaxis, :, self._lskp:self._rskp]

        imgs = None
        for file in self.rsd_list:
            img = np.fromfile(file, dtype='<i2')
            img = img.reshape(self._blk, self._Nx, self._Ny)[..., self._lskp:self._rskp]
            img = background + img
            img = img.astype(np.uint16)
            if imgs is None:
                imgs = img
            else:
                imgs = np.concatenate([imgs, img])
        return imgs[start_frame:end_frame:step]
    
    def get_metadata(self):
        """Returns the metadata dictionary"""
        return self._meta