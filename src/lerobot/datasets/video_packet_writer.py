#!/usr/bin/env python3

# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Direct video packet writing for pre-compressed video streams.

This module provides functionality to write pre-encoded video packets (e.g., from ROS2 FFMPEGPacket messages)
directly to MP4 files without decoding/re-encoding. This significantly improves recording performance
and preserves original video quality.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import av

logger = logging.getLogger(__name__)


def normalize_codec_name(codec: str) -> str:
    """
    Normalize encoder/codec names to PyAV-compatible codec names.

    Args:
        codec: Codec name from source (e.g., "hevc_vaapi", "h265", "h264")

    Returns:
        Normalized codec name for PyAV (e.g., "hevc", "h264")
    """
    codec_lower = codec.lower()

    # # SHR ADD: Map various AV encoder names to 'libsvtav1'
    # if any(name in codec_lower for name in ["hevc_vaapi"]):
    #     return "libsvtav1"

    # Map various HEVC/H.265 encoder names to "hevc"
    if any(name in codec_lower for name in ["hevc", "h265", "x265" ,"hevc_vaapi"]):
        return "hevc"

    # Map various H.264/AVC encoder names to "h264"
    if any(name in codec_lower for name in ["h264", "avc", "x264"]):
        return "h264"

    # Return as-is if already normalized or unknown
    return codec


@dataclass
class VideoPacketInfo:
    """Information about a video packet."""

    width: int
    height: int
    codec: str  # e.g., "hevc", "h264"
    pix_fmt: str = "yuv420p"  # Pixel format


class VideoPacketWriter:
    """
    Writes pre-encoded video packets directly to MP4 files using PyAV.

    This writer bypasses the normal encode step by directly muxing compressed packets
    into MP4 containers. This is ideal for video streams that are already compressed
    (e.g., from hardware encoders or ROS2 FFMPEGPacket messages).

    Example:
        ```python
        # Create writer
        writer = VideoPacketWriter(
            video_path="output.mp4",
            width=640,
            height=480,
            fps=30,
            codec="hevc"
        )

        # Write packets
        for packet_data, pts, flags in packet_stream:
            writer.write_packet(packet_data, pts, flags)

        # Finalize
        writer.close()
        ```
    """

    def __init__(
        self,
        video_path: Path | str,
        width: int,
        height: int,
        fps: int,
        codec: str = "hevc",
        pix_fmt: str = "yuv420p",
    ):
        """
        Initialize video packet writer.

        Args:
            video_path: Output MP4 file path
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            codec: Video codec (e.g., "hevc", "h264")
            pix_fmt: Pixel format (e.g., "yuv420p")
        """
        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)

        self.width = width
        self.height = height
        self.fps = fps
        # Normalize codec name (e.g., "hevc_vaapi" → "hevc", "h265" → "hevc")
        self.codec = normalize_codec_name(codec)
        self.pix_fmt = pix_fmt

        # Open container and create stream
        self.container = av.open(str(self.video_path), "w")

        # Add video stream for muxing pre-encoded packets
        # The rate parameter sets the stream's time_base automatically
        self.stream = self.container.add_stream(self.codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self.stream.pix_fmt = pix_fmt

        # Calculate PTS increment per frame based on time_base
        # PyAV uses time_base = 1/15360 by default
        # PTS increment = time_base_denominator / fps
        self._time_base_den = 15360
        self._pts_per_frame = self._time_base_den // fps

        self._packet_count = 0
        self._closed = False

        logger.debug(f"VideoPacketWriter initialized for {self.video_path} (codec={codec}, {width}x{height}@{fps}fps, pts_per_frame={self._pts_per_frame})")

    def write_packet(self, packet_data: bytes, pts: int, flags: int = 1) -> None:
        """
        Write a pre-encoded video packet to the file.

        Args:
            packet_data: Raw compressed packet data (bytes)
            pts: Presentation timestamp (ignored - we use frame index instead)
            flags: Packet flags (bit 0 = keyframe)

        Raises:
            RuntimeError: If writer is closed
        """
        if self._closed:
            raise RuntimeError("Cannot write to closed VideoPacketWriter")

        # Create PyAV packet from raw data
        # Using the documented pattern from PyAV GitHub issue #216
        packet = av.Packet(len(packet_data))
        packet.update(packet_data)

        # Set timing information
        # IMPORTANT: Scale packet_count to match stream's time_base
        # time_base = 1/15360, so for 30fps, each frame = 512 time units
        scaled_pts = self._packet_count * self._pts_per_frame
        packet.pts = scaled_pts
        packet.dts = scaled_pts
        packet.stream = self.stream

        # Set keyframe flag based on AVPacket flags (AV_PKT_FLAG_KEY = 0x0001)
        if flags & 0x01:
            packet.is_keyframe = True
        else:
            packet.is_keyframe = False

        # Mux the packet directly (no encoding!)
        try:
            self.container.mux(packet)
            self._packet_count += 1
        except Exception as e:
            logger.error(f"Error muxing packet: {e}")
            raise

    def close(self) -> None:
        """Finalize and close the video file."""
        if self._closed:
            return

        try:
            self.container.close()
            self._closed = True
            logger.debug(f"VideoPacketWriter closed. Wrote {self._packet_count} packets to {self.video_path}")

            # Verify file was created
            if not self.video_path.exists():
                raise IOError(f"Video file was not created: {self.video_path}")

        except Exception as e:
            logger.error(f"Error closing VideoPacketWriter: {e}")
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class VideoPacketBuffer:
    """
    Buffers pre-encoded video packets during episode recording.

    This class accumulates packets from multiple cameras during an episode,
    then writes them all to MP4 files when the episode is saved.

    Example:
        ```python
        buffer = VideoPacketBuffer(root_dir="dataset", fps=30)

        # During recording
        buffer.add_packet("camera_left", packet_msg)
        buffer.add_packet("camera_right", packet_msg)

        # Save episode
        buffer.save_episode(episode_index=0)
        ```
    """

    def __init__(self, root_dir: Path, fps: int):
        """
        Initialize packet buffer.

        Args:
            root_dir: Root directory for the dataset
            fps: Frames per second for video
        """
        self.root_dir = Path(root_dir)
        self.fps = fps

        # Buffer: {camera_name: [packet_data, ...]}
        self.packets = defaultdict(list)

        # Video info: {camera_name: VideoPacketInfo}
        self.video_info = {}

    def add_packet(
        self, camera_name: str, packet_data: bytes, 
        width: int, height: int, codec: str, 
        pts: int=0, flags: int=1
    ) -> None:
        """
        Add a packet to the buffer.

        Args:
            camera_name: Name of the camera
            packet_data: Raw packet bytes
            width: Video width
            height: Video height
            codec: Codec name (e.g., "hevc", "h264")
            pts: Presentation timestamp. (Optional)
            flags: Packet flags, 1 is key_frame. (Optional)
        """
        # Store video info from first packet
        if camera_name not in self.video_info:
            # Normalize codec name (e.g., "hevc_vaapi" → "hevc", "h265" → "hevc")
            normalized_codec = normalize_codec_name(codec)
            self.video_info[camera_name] = VideoPacketInfo(
                width=width, height=height, codec=normalized_codec, pix_fmt="yuv420p"
            )

        # Buffer packet data
        self.packets[camera_name].append({"data": packet_data, "pts": pts, "flags": flags})

    def delete_final_packet(self, camera_name: str) -> None:
        if camera_name in self.packets and len(self.packets[camera_name]) > 0:
            self.packets[camera_name].pop()

    def save_episode(self, episode_index: int, dataset_meta) -> None:
        """
        Write all buffered packets to video files for an episode.

        Args:
            episode_index: Index of the episode
            dataset_meta: Dataset metadata object with get_video_file_path() method
        """
        # packet_list_length = 0
        for camera_name, packet_list in self.packets.items():
            if len(packet_list) == 0:
                continue

            info = self.video_info[camera_name]

            # Get correct video path from dataset metadata
            # IMPORTANT: Use full video_key (e.g., "observation.images.left_color"), not just camera_name
            # This matches what encode_episode_videos() expects
            video_key = f"observation.images.{camera_name}"
            video_path = self.root_dir / dataset_meta.get_video_file_path(episode_index, video_key)

            logger.info(f"Saving {len(packet_list)} packets to {video_path}")
            # if packet_list_length == 0:
                # packet_list_length = len(packet_list)

            # Create writer and mux all packets
            writer = VideoPacketWriter(
                video_path=video_path,
                width=info.width,
                height=info.height,
                fps=self.fps,
                codec=info.codec,
                pix_fmt=info.pix_fmt,
            )

            for packet in packet_list:
                writer.write_packet(packet["data"], packet["pts"], packet["flags"])

            writer.close()
        # return packet_list_length

    def clear(self) -> None:
        """Clear packet buffer."""
        self.packets.clear()

    def get_packet_count(self, camera_name: str) -> int:
        """
        Get number of packets buffered for a camera.

        Args:
            camera_name: Name of the camera

        Returns:
            Number of buffered packets
        """
        return len(self.packets.get(camera_name, []))
    
    @property
    def episodeLength(self) -> int:
        """
        Returns:
            Number of buffered packets for the first camera (assuming all cameras have the same number of packets)
        """
        if not self.packets:
            return 0
        else:
            for _, packets_ls in self.packets.items():
                return len(packets_ls)
