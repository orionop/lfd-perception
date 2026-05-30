# config
Most [units] depend on the input data source, but typical units are provided.

## "preprocessing"
### "desired_sample_time"
float
Sample time at which data is extracted from the data source.

## "input"
### "path"
string
Path to file to be processed. Must have file extension.
Can double as write path when output path not specified.

### "data_identifiers"
identifier in the to be loaded data file, to map data. 
How these configs are used depends on the data loader function.
For example, this could be column names in a csv file.
The "rotation_w" is for cases where rotations are parameterized by 4 numbers such as quaternions.

### "rotation_format"
Indicating how rotations are stored in the data. Used to select a particular processing method, as the pose input to ReFrameId must be a 4x4 homogeneous transformation matrix. One of the following:
- quaternions
- <TODO>

## "output"
### "path"
string - optional
Path to which .pickle with task model is written.
When not specified, path and filename are derived from input path.

## "identification"
### "method"
string
Method to be used in the frame identification.
* "twist"
* "wrench"
* "power"

### "max_iter"
int
Max number of iterations for the optimizer.

### "norm_time"
scalar
Norm level used over samples in the data window.
Two-norm is root-mean-square.

### "norm_axes"
scalar or string
Norm level used over the axes in the data window, after the time norm.
Can be scalar or string. When string, special case is used, see documentation of tools/calc/normal_p_norm.

### "frame_types_rotation"
list of strings
Frame types, either ground or body, to consider in optimization for finding rotation-only interaction frames.
* "G"
* "B"

### "frame_types_translation"
list of strings
Frame types, consisting of either ground or body, to consider in optimization for finding translation-only interaction frames. First letter corresponds to rotation component, second letter corresponds to translation component.
* "GG"
* "BB"
* "GB"
* "BG"

### "frame_types_combined"
list of strings
Frame types, consisting of either ground or body, to consider in optimization for finding rotation and translation combined interaction frames. First letter corresponds to rotation component, second letter corresponds to translation component.
* "GG"
* "BB"
* "GB"
* "BG"

### "init_random"
bool
If true, randomly initialize the intial conditions for the optimization.
If false, use the initial conditions specified below.

### "init_angles"
list of 3 scalars, in [m]
Optimizer interaction frame initial angles for rotation frame identification.

### "bound_angles"
scalar, in [m]
Optimizer bounds on angles for rotation frame identification.

### "init_origin"
list of 3 scalars, in [rad]
Optimizer interaction frame initial origin for rotation frame identification.

### "bound_origin"
scalar, in [rad]
Optimizer bounds on origin for rotation frame identification.

### "bound_origin_combined_ground_offset"
list of 3 scalars, in [m]
Experimental. Only for combined method (twist + wrench).
Adds an offset to the initial position and the bounds for frames with an origin in ground. This helps to find a frame that is not located in the origin of the robot, which may be far away from where the actual interactions are happening. Expressing control in frames far away may lead to instabilities / sensitivities. This may help prevent that. Essentially adding prior knowledge

## "segmentation_classification"
### "thresholds"
Thresholds (absolute) above which something is considered to move or exert a force. Used to segment data, and also classify axes of the interaction frame into interaction types.

#### "angular_velocity"
scalar, in [rad/s]

#### "linear_velocity"
scalar, in [m/s]

#### "moment"
scalar, in [Nm]

#### "force
scalar, in [N]

### norm_time
scalar, in [s]
Norm level used over samples in the segment. Classes are determined over the norm of the samples over the segment window.
Two-norm is root-mean-square.

### dead_time
scalar, in [s]
The time within which it is not allowed for a new segmentation boundary to occur with respect to the preceding segmentation. 

### used_signals
list of 4 booleans
Signals used for segmentation.
First entry corresponds to angular velocities.
Second entry corresponds to linear velocities.
Third entry corresponds to moments.
Fourth entry corresponds to forces.

### "direction"
string
Either "unilateral" or "bilateral"
Indicating whether axes of the identified frame(s) are classified on interaction type in a unilateral or bilateral way. In the bilateral case, it may be possible to distinguish different interaction types on the same axis. For example, pushing on a surface in one direction, while the opposed direction on the same axis is free.

### "gripper_velocity_threshold"
scalar, in [m/s]
Minimal velocity for which the gripper fingers are considered to be moving.

### gripper_dead_time
scalar, in [s]
See "dead_time", but for gripper

### frame_offset
scalar, in [m]
Experimental. To deal with certain edge cases