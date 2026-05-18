import csv
import os
import json
from datetime import datetime, timezone

import cv2
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

self_id = os.environ['SELF_ID']
with open("parameters.json", 'r') as f:
    parameters = json.load(f)


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')

        # --- Parameters ---
        self.declare_parameter('image_topic', f'car_{self_id}/image_raw/compressed')
        #self.declare_parameter('output_dir', 'images_out')
        self.declare_parameter('image_format', 'png')       # 'jpg' or 'png'
        self.declare_parameter('save_every_n', 2)           # save every Nth frame
        self.declare_parameter('use_header_stamp', True)    # filenames use header.stamp
        #self.declare_parameter('csv_name', 'images_log.csv')
        self.declare_parameter('overwrite_existing_csv', False)
        
        current_time = datetime.now()

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.output_dir = parameters["save_folder"] #self.get_parameter('output_dir').get_parameter_value().string_value
        self.image_format = self.get_parameter('image_format').get_parameter_value().string_value.lower()
        self.save_every_n = int(self.get_parameter('save_every_n').get_parameter_value().integer_value)
        self.use_header_stamp = bool(self.get_parameter('use_header_stamp').get_parameter_value().bool_value)
        self.csv_name = f'images_only_{current_time.strftime("%Y%m%d-%H%M")}.csv' #self.get_parameter('csv_name').get_parameter_value().string_value
        self.overwrite_csv = bool(self.get_parameter('overwrite_existing_csv').get_parameter_value().bool_value)

        if self.image_format not in ('jpg', 'png', 'jpeg'):
            self.get_logger().warn(f"Unsupported image_format '{self.image_format}', defaulting to 'jpg'")
            self.image_format = 'jpg'

        # Prepare filesystem
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = os.path.join(self.output_dir, self.csv_name)

        # CSV handling
        write_header = self.overwrite_csv or (not os.path.exists(self.csv_path))
        self.csv_file = open(self.csv_path, 'w' if self.overwrite_csv else 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        if write_header:
            self.csv_writer.writerow([
                'filename',
                'header_stamp_sec',
                'header_stamp_nanosec',
                'received_walltime_iso8601'
            ])
            self.csv_file.flush()

        # Bridge and counters
        self.bridge = CvBridge()
        self.frame_count = 0
        self.saved_count = 0

        # QoS tuned for camera topics
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            qos
        )

        self.get_logger().info(
            f"Listening on '{self.image_topic}'. Saving images to '{self.output_dir}' "
            f"as .{self.image_format}; logging to '{self.csv_path}'."
        )

    def image_cb(self, msg: Image):
        self.frame_count += 1
        if self.save_every_n > 1 and (self.frame_count % self.save_every_n != 0):
            return

        # Timestamps
        stamp_sec = int(msg.header.stamp.sec)
        stamp_nanosec = int(msg.header.stamp.nanosec)
        now_iso = datetime.now(timezone.utc).astimezone().isoformat()

        # Filename
        if self.use_header_stamp:
            base = f"img_{stamp_sec:010d}_{stamp_nanosec:09d}"
        else:
            # wall time based name to avoid collisions if header not populated
            tnow = datetime.now(timezone.utc)
            base = tnow.strftime("img_wall_%Y%m%d_%H%M%S_%f")

        filename = f"{base}.{self.image_format}"
        fpath = os.path.join(self.output_dir, filename)

        # Convert & save
        try:
            # Try to preserve encoding if already 8-bit; fallback to BGR8
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        ok = cv2.imwrite(fpath, cv_img)
        if not ok:
            self.get_logger().error(f"Failed to write image: {fpath}")
            return

        # CSV row
        self.csv_writer.writerow([filename, stamp_sec, stamp_nanosec, now_iso])
        self.csv_file.flush()

        self.saved_count += 1
        if self.saved_count % 50 == 0:
            self.get_logger().info(f"Saved {self.saved_count} images so far…")

    def destroy_node(self):
        try:
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.flush()
                self.csv_file.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = ImageSaverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

