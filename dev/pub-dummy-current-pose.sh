#!/bin/bash -e

DOMMY=('{
header: {
    seq: 0,
    stamp: {
      secs: 0,
      nsecs: 0
    },
    frame_id: ''
  },
pose: {
    position: {
      x: 797.668,
      y: 1129.12,
      z: 0.0
    },
    orientation: {
      x: -0.0
      y: 0.0
      z: 0.00788880626314
      w: -0.999968882884
    }
  }
}')
echo $DUMMY
rostopic pub /current_pose geometry_msgs/PoseStamped "${DUMMY}"
