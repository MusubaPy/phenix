### Bioinspiration &

### Biomimetics^

PAPER

## A theorem of target points for the ground reaction

## force in a planar serial-linked walking limb

To cite this article: Alexander N Kuznetsov 2018 Bioinspir. Biomim. 13 066010

View the article online for updates and enhancements.

### You may also like

```
A wearable force plate system for the
continuous measurement of triaxial ground
reaction force in biomechanical
applications
Tao Liu, Yoshio Inoue and Kyoko Shibata
```
-

```
Towards a bio-inspired leg design for high-
speed running
Arvind Ananthanarayanan, Mojtaba Azadi
and Sangbae Kim
```
-

```
Evidence for multiple dynamic climbing
gait families
Jason M Brown, Max P Austin, Bruce D
Miller et al.
```
-

```
This content was downloaded from IP address 83.69.192.106 on 17/09/2025 at 13:
```

```
© 2018 IOP Publishing Ltd
```
### List of symbols

```
( y ) The line through the point O of limb contact with
the ground, along which the ground reaction
force acts with zero net mechanical power, so that
no external work is done by the limb; in steady
legged locomotion this line is kept close to vertical
( x ) The line through the point of limb contact
with the ground, which is perpendicular to the
line ( y ); together they represent the axes of
Cartesian coordinates
O The distal end-point of the limb (foot, paw or
hoof ) which makes contact with the ground; here
it is taken as the origin of Cartesian coordinates
```
```
i Any limb joint, specified by capital letters
except O which is reserved for the limb end-
point, and T reserved for Alexander’s target
point
A , B The joints of an abstract two-segment limb
P , M , D The proximal, middle, and distal joints of the
three-segment Z -like limb, respectively, for
example, hip, knee, and ankle, or scapular
apex, shoulder joint, and elbow
T Alexander’s target point for ground reaction
forces
xi , yi Instantaneous positions of a limb joint i
against Cartesian axes formed by lines ( x )
and ( y )
```
A N Kuznetsov

066010

BBIICI

© 2018 IOP Publishing Ltd

13

Bioinspir. Biomim.

BB

1748-

10.1088/1748-3190/aae44f

6

### 1

### 17

Bioinspiration & Biomimetics

IOP

### 19

### October

# A theorem of target points for the ground reaction force in a planar

# serial-linked walking limb

```
Alexander N Kuznetsov
Borissiak Paleontological Institute, Russian Academy of Sciences, Moscow 117647, Russia
E-mail: sasakuzn@mail.ru
Supplementary material for this article is available online
Keywords: legged locomotion, multi-actuator limb, inter-actuator antagonism, mechanical work minimization, optimum kinematics,
optimum force direction
```
#### Abstract

#### An important energy expense in the legged locomotion of both animals and robots is the mechanical

#### antagonism between muscles/actuators, which is the positive mechanical work of some muscles

#### opposed by the simultaneous negative work of the others. One known way to minimize the

#### mechanical antagonism is the proper employment of the redundant degrees of freedom of the

#### limb. Here, I present and analyze a generalized model of a planar serial-linked limb composed of

#### any number of segments and conclude that the minimization of the inter-actuator antagonism

#### requires fixation of all the joint angles except the two defined by simple geometric considerations.

#### So, regardless of the number of joints, the limb should optimally act as a two-joint system. Which

#### of the redundant joints to fix or move depends on the instantaneous position of the joints relative

#### to the vertical line through the center of the foot contact with the ground. Subsequently, for the

#### first time, I pose and solve the following problem: how to eliminate the inter-actuator antagonism

#### by the adjustment of the horizontal component of the ground reaction force. The solution is that,

#### during the contact phase, the ground reaction force vector should be redirected so as to maintain

#### alignment with the limb joints in the order in which they attain the smallest angular deflection

#### from the vertical line through the foot. As the joints pass the vertical line through the point of limb

#### contact with the ground one by one, abrupt changes of the horizontal ground force component from

#### positive to negative should occur. Mammals cannot follow this algorithm exactly due to muscular

#### actuator limitations and they tend to align the ground reaction force with some compromise target

#### point above the hip or scapula. The suggested principles of the optimal choice of redundant degrees

#### of freedom and direction of the ground reaction force can be implemented in robotics to achieve a

#### lower cost of transport than is known for animals.

##### PAPER

2018

```
RECEIVED
17 June 2018
REVISED
21 August 2018
ACCEPTED FOR PUBLICATION
26 September 2018
PUBLISHED
19 October 2018
```
```
Bioinspir. Biomim. 13 (2018) 066010 https://doi.org/10.1088/1748-3190/aae44f
```

_ωi_ Instantaneous angular velocity in a
joint _i_ ; the joint angles are measured
at the rear side of the limb
_v_ trunk velocity relative to the
ground; it is taken to be kept
horizontal and the same for all
trunk parts (pitching and spine
bending are dismissed)
_vA_ , _vB_ , ..., _vZ_ circumferential velocity vectors
of the distal limb point _O_ due to
rotation in individual joints
_h_ ( _y_ )-wise dimension (i.e. height)
of the contour of the vector
summation at the limb end-point
_O_ of circumferential velocities
produced by angular movements
in individual limb joints; in the
two-segment leg, when the trunk
velocity is horizontal, _h_ = 2| _P_^0 |/ _Fy_ ;
this is the geometric representation
of the mechanical antagonism
between the muscles/actuators of
different joints
_hA_ , _B_ , _h_ A,C, _hB_ , _C_ ... Heights of parallelograms of the
vector summation of circum-
ferential velocities produced
by angular movements in every
possible two-combinations of the
limb joints
_θ_ Angle of the instantaneous ground
reaction force _F_ from the vertical
( _y_ ) axis
_θi_ Angle of the limb joint _i_ (or, more
exactly, of the radius-vector _Oi_ )
from the vertical ( _y_ ) axis
_F_ Instantaneous ground reaction
force
_Fy_ Component of the instantaneous
ground reaction force acting along
the line ( _y_ ); in steady-legged
loco motion it is vertical and
counteracts gravity acting upon
the body
_Fx_ Component of the instantaneous
ground reaction force acting
perpendicular to the line ( _y_ ); in
steady-legged locomotion it causes
cyclic fluctuations of the horizontal
velocity of the body; in the current
model it is treated as the variable for
energetic optimization
_FxA_ , _FBx_ The values of the _Fx_ component of
the instantaneous ground reaction
force, which ensure alignment
of the resultant force _F_ vector
with joint _A_ or _B_ , respectively, of
an abstract two-segment limb,
provided that its _Fy_ component is
given as constant

```
P limb Total mechanical power produced by the
limb at a given instant of the contact phase
PA , PB Mechanical power produced instantaneously
in joint A or B , respectively, of an abstract
two-segment limb
PAB Mechanical power produced instantaneously
in joint B of an abstract two-segment limb
when the ground reaction force F is aligned
with the other joint A ( Fx = FAx )
PBA mechanical power produced instantaneously
in joint A of an abstract two-segment limb
when the ground reaction force F is aligned
with the other joint B ( Fx = FBx )
P^0 Mechanical power produced in one of the
joints of an abstract two-segment limb when
the instantaneous direction of the ground
reaction force coincides with the line ( y ), i.e.
when Fx = 0
```
### Color scheme in the figures

```
Blue Ground reaction force vector, or its
theoretically predicted line of action (see
the supplementary video, available online at
stacks.iop.org/BB/13/066010/mmedia), or
its vertical component Fy (figure 2(C)) and
respective joint powers P^0 and − P^0 (figure 5)
Magenta The joints which are instantaneously the
best to move
Red The best target joint for the ground
reaction force, and respective horizontal
component of this force (figure 2(C)), and
respective power produced in the other
joint (figure 5)
Green The worse target joint for the ground
reaction force, and respective horizontal
component of this force (figure 2(C)), and
respective power produced in the other
joint (figure 5)
```
### 1. Introduction

```
Recently, the biomechanical fundamentals of animal
locomotion have acquired applied importance due
to the outstanding progress in robotic designs. At the
dawn of robotics, the control and sensing of motion
were considered more important problems than the
structural optimization of the limb hardware. As a
result, current walking robots generally consume
considerably more energy than similar-sized animals,
with rare exceptions, for example, the MIT cheetah
quadrupedal robot (Seok et al 2013). Equipped
with mammalian-like limbs, the MIT cheetah
robot approaches the overall efficiency of running
mammals due to the usage of electric actuators which
are much more efficient in the conversion of internal
energy (electric or metabolic) to external mechanical
power than animal muscles. Why is the MIT cheetah
robot not more efficient than quadrupedal animals
```

if its actuators are better? This could be because the
MIT cheetah robot does not employ some other
mechanisms of energy economy which are, in contrast,
used by animals. One can say that the robot is not
sufficiently biomimetic to be more energy efficient
than quadrupedal animals.
What did animal evolution do better with mechan-
ical limb design and usage than robotic designers?
I propose, and will attempt to confirm below, that
the answer is connected to the different utilizations
of multiple muscles/actuators where multiplicity
is a common feature of animal and robot limbs. The
multiple muscles/actuators can come into antagonis-
tic counteractions with each other. This is not due to
the presence of anatomical antagonists (for example,
the flexor and extensor of the same joint), but due to
the simultaneous activity of muscles/actuators of dif-
ferent joints during which some of them perform
positive, and the others perform negative (braking),
mechanical work. This useless waste of energy should
be avoided but sometimes it is impossible. The most
striking example is the steady-speed locomotion over
level ground with serial-linked limbs like those of ani-
mals, the MIT cheetah and many other robots. The
limbs cannot act like a wheel because, generally, the
ground reaction force is aligned with only one of its
joints at a time, while the other ones must be balanced
actively by muscular/actuator torques. This activity,
in turn, is associated with mechanical work because
the joint angles inevitably change, and the actuators
respectively rearrange geometry (for example, muscles
shorten or lengthen). This mechanical work is exces-
sive as compared to the wheel performance. Positive
power is developed by actuators of some joints (for
example, by shortening muscles), as in the wheel of an
accelerating vehicle, and negative power is developed
by actuators of the other joints (for example, by mus-
cles being forcibly stretched), as in the wheel of a decel-
erating vehicle, even though the leg neither accelerates
nor decelerates the animal or robot when it acts in a
wheel-like manner. Unfortunately, the phenomenon
of such an antagonism has been insufficiently stud-
ied in robotics, and even less so in biomechanics. To
my knowledge, the problem was first posed in human
biomechanics by Elftman (1939, 1940), then in animal
biomechanics by Alexander (Alexander and Vernon
1975, Alexander 1976, 1977), and finally in robotics by
Waldron and Kinzel (1981). Then, kinematics optim-
ization as a means to minimize the antagonism power
losses was introduced. This was first conducted in the
biomechanics of mammalian limbs by the author
(Kuznetsov 1995), who showed a reduction of the
absolute magnitude of mechanical work by the use of
a redundant degree of freedom of the legs. Recently, a
more general metrics of the antagonism phenomenon
was developed in a few robotics studies (Abate _et al_
2016, Cahill _et al_ 2017). They show that, not only par-
ticular kinematics, but the general limb structure, can
be treated as a way to reduce the inter-actuator power

```
antagonism when performing a mechanical task for
which the limb is designed. In fact, this approach by
robotics researchers can be very helpful in biomechan-
ics too. It paves the way for understanding the struc-
tural variety of terrestrial animal limbs from a unified
point of view because the antagonism is a fundamen-
tal energetic problem for any multi-actuator limb.
So, each particular limb structure can be potentially
regarded as the unique solution to the antagonism
minimization problem with respect to the unique bio-
logical task of this particular limb.
Robotics researchers (Abate et al 2016, Cahill et al
2017) define the limb task specifically as the typical
trajectory of the distal limb point (the end effector)
relative to the vehicle trunk associated with typi-
cal external forces (for example, the ground reaction
forces) applied to the end effector in every point of
this trajectory. Here, I will loosen this definition for
the case of the limb contact phase with the ground
in steady-state walking or running. In this case, the
task of the limb is to counteract the gravity acting
on the body, and so only the vertical component of
the ground reaction force can be regarded as neces-
sary for the task. Then, the horizontal component of
this force can be taken as a free variable which can be
employed in the antagonism minimization (Alexan-
der 1991). So, I will attempt to optimize the horizontal
component of the ground reaction force, and hence,
the direction of the total ground reaction force, for
the constant-speed horizontal locomotion over even
terrain. I will consider simple serial-linked limbs, as
in animals, namely, the simplest of them—the planar
(parasagittal) limbs. In the animal kingdom, almost
all such legs, namely the fore and hind legs of theri-
ans (i.e. placental and marsupial mammals) and the
hind legs of birds as well as many non-avian dino-
saurs, have acquired, in the course of evolution, the
same basic structure. They are composed of three seg-
ments arranged in the parasagittal plane as a Z -like
zigzag, with the upper bend pointing forward. These
segments are scapula, humerus and antebrachium in
the therian forelimb, and femur, crus and pes in the
hind limb (Kuznetsov 1985, Fischer 1998). The legs of
the MIT cheetah robot superficially mimic the Z -like
zigzags of the natural cheetah legs but do not employ
their potential advantages adequately.
In robotics, the task of the optimization of the
ground reaction force components in order to mini-
mize energy consumption is not new (for example, Kar
et al (2001) and Agarwal et al (2012)). In fact, in the
articles cited, the inter-actuator antagonism is mini-
mized, although it is not named so. But the problem
of such computational models is the lack of theor-
etical appreciation of the optimal solutions found by
a computer search. In this respect, the analytic solu-
tions are preferable. Some general analytic approaches
were developed in biomechanics, but they do not take
into account the mechanical antagonism between the
actuators of the same limb. Typically, they explore the
```

minimization of the mechanical antagonism between
the different limbs, the net mechanical work of each
limb being taken as a whole (Alexander 1980, Donelan
_et al_ 2002). The examples of deeper, joint-by-joint or
muscle-by-muscle analysis are still rather rare but very
promising (Prilutsky _et al_ 1996).
The problem is that there is no ground reaction
force theory yet which allows one to predict solu-
tions with minimal inter-actuator antagonism in the
limb. The experimental studies of ground reaction
forces in terrestrial animals are numerous, but gener-
alizations of these data are few. Usually, the temporal
changes of the vertical and horizontal components of
the ground reaction forces are approximated as wave-
forms by a Fourier series (Alexander and Jayes 1980).
Although powerful, this method is superficial, and it
is problematic to determine the optimal force profile
in this way. Even if based on some simple principle,
this profile would most probably be described by a
long and awkward Fourier series which would not
help in understanding the principle and applying it
in robotics.
The most unexpected generalization on the ground
reaction forces in animals was drawn from experi-
ments on dogs and sheep by Jayes together with Alex-
ander who was guiding the research of terrestrial loco-
motion biomechanics in the last quarter of the 20th
century. Jayes and Alexander (1978) demonstrated
that, irrespective of speed and gait, the force exerted by
a paw or hoof on the ground changes direction during
the contact phase of each particular leg so as to remain
in line with a point fixed relative to the animal’s trunk
and located well above the limb apical pivot (figure 1),
i.e. the acetabulum or the scapular fulcrum located at
its dorsal end whereupon the slips of the serratus ven-
tralis muscle converge. Before this finding, Alexander
thought that the ground reaction force in steady walk-
ing and running should be kept aligned with the limb

```
apical pivot itself, not with some abstract point in the
space above it. I suggest that it is referred to as ‘Alexan-
der’s point’. In fact, the existence of such a point means
that the vertical (directed upward to counteract grav-
ity) and longitudinal (braking-to-propulsive) comp-
onents of the ground reaction force change during the
contact phase in a coupled and specifically constrained
manner. The remarkable feature of this coupling is that
the total ground reaction force is always more vertical
than the limb itself, and that it deviates from the limb
proximo-distal axis according to a rather simple rule.
So, the longitudinal component can be derived from
the vertical one, and vice versa. Based on this rule,
Alexander and Jayes assumed, in a series of subsequent
publications, a simple time-dependent proportional-
ity between the vertical and horizontal components of
the ground reaction force, but discovered that this pro-
portionality is not sufficiently compatible with Fourier
approximation (Alexander and Jayes 1980). The prob-
lem was not researched further, so the phenomenon
of Alexander’s point did not obtain any theoretical
explanation and was abandoned in the biomechanics
of quadrupeds.
Thirty years later, a similar point was theoretically
considered in the biomechanics of bipeds, especially
in modeling human orthogradism (Maus et al 2010,
Blickhan et al 2015). The idea is that the human trunk,
due to its upright position, has a center of mass well
above the limb apex, i.e. the hip joint, and so appears
to be unstable. It could gain stability if it was somehow
suspended to a virtual pivot point placed well above
the center of mass of the trunk as if it was seated in a
rocking chair, and, like the rocking chair, the ground
reaction forces produced at different stages of the
motion cycle should converge to this point above the
trunk, and the trunk should turn about this center
although it is not shaped as a material structure. That is
why it is termed ‘pivot’ and why it is termed ‘virtual’.
Note also that ‘virtual pivot point’ is a synonym for the
‘instantaneous center of rotation’ of a rigid body mov-
ing along a curved path. All the above considerations
of the trunk equilibrium problem in human ortho-
grade bipedalism is inapplicable to a dog, sheep, and
any other quadrupedal therian mammal, since their
trunk hangs down from the limbs’ apices (points P in
figure 1). So, Alexander’s point hardly has the nature of
a virtual pivot point and is definitely not an instanta-
neous center of rotation of the trunk because, accord-
ing to Jayes and Alexander (1978), its position is kept
in both pitching (gallop) and non-pitching (walk, trot)
gaits. I will attempt to re-evaluate Alexander’s point
in a quite different respect, namely as a way to mini-
mize muscular activity by exclusion of the mechanical
antagonism of the muscles of different joints within
the same limb.
Generally, there are two major parameters of
muscular activity, the minimization of which is use-
ful in steady legged locomotion—the muscular force
and the muscular work. Each one of them has its
```
```
Figure 1. Target points (asterisks T ) for ground reaction
forces (blue vectors) in a dog according to figure 12(a)
of Jayes and Alexander (1978), with major limb joints
superimposed. Proximal articulations (triangles P ), middle
joints (circles M ) and distal joints (squares D ) are the hip,
knee, and ankle, or scapular pivot, shoulder joint, and
elbow. O —contact points of the limbs with the ground.
```

own metabolic cost, and each one can be reduced by
optim ization of the ground reaction force direction.
The muscular force is minimized through the mini-
mization of joint moments (torques), more exactly
of the sum of absolute magnitudes of moment arms
of the ground reaction force relative to the leg joints^1.
In its turn, the muscular work is reduced through the
minimization of the sum of the absolute magnitudes
of products of the same moment arms of the ground
reaction force with angle changes in respective joints.
So, by these equations, the muscular force and mus-
cular work minima may not coincide with each other.
Although successful in solving some problems, such as
a transition from crouched-to-upright limb posture
with body mass growth in mammals (Biewener 1983,
1989) and straight-knee heel-down human walking
(Günther _et al_ 2004), the joint moments’ minimiza-
tion can hardly explain the existence of Alexander’s
point. To obtain the joint moments’ minimum, the
best target point for the ground reaction force should
be placed just in the hip joint (zero moment arm rela-
tive to the hip, as in the model by Günther _et al_ (2004)),
not above it, and this condition should be kept from the
beginning of the contact phase up to the end. Indeed,
even a small forward deviation of the ground reaction
force from the hip would decrease the moment arm
of this force in one joint (the knee) but increase the
moment arms in the other two (the ankle and the hip
itself ), and any backward force deviation from the hip
would lead to a similar result (the changes of moment
arms in the knee and the ankle would be reversed, but
the absolute magnitude of the moment arm in the hip
would be increased again). Note also that the increase
of the moment arm in the hip alone would be greater
than the respective decrease in either the knee or the
ankle, because (1) the hip is farther from the point
of foot contact with the ground, around which the
ground reaction force vector is rotated, and (2) for a
given angular deflection of this vector, the moment
arms change faster if they start from zero because they
change with the cosine of the angle (the arm itself being
proportional to the sine). It follows that Alexander’s
point cannot be explained in terms of muscular force
minimization, and I will concentrate on the second
opportunity—the minimization of muscular work.

```
The model which I will develop below can be clas-
sified among the so-called ‘collisional models’ of leg-
ged locomotion (Ruina et al 2005, Lee et al 2011). They
treat the limb contact with the ground as a col lision
which redirects the velocity of the body. When the
ground reaction force is perpendicular to the velocity,
there is no collision, and no mechanical work is done
to redirect the body. So, the work is associated with the
force component, which is aligned with the velocity.
The instantaneous collisional work (the power) is cal-
culated as a dot product of the ground reaction force
vector and the velocity vector of the center of mass of
the body. The advantage of collisional models is that
they do not initially rely on any energy recovery mech-
anisms (such as the elastic recoil of tendons or pendu-
lar effects), not excluding them as the secondary means
to reduce energy expenditure. By now, two kinds of
collisional models have been employed: (1) the ‘com-
bined limbs method’ based on the total ground reac-
tion force produced collectively by all the limbs which
are currently in touch with the ground (Lee et al 2011),
and (2) the ‘individual limbs method’ based on a sepa-
rate analysis of the ground reaction force of every limb
(Donelan et al 2002). The former method is aimed at
broader though rather superficial analysis. The latter
method is aimed at more in-depth analysis and allows
the inter-limb antagonism to be distinguished, which
arises when one limb performs negative mechanical
work, and, at the same time, the other limb performs
positive mechanical work, thus increasing the col-
lision of the body against the first one. I use the third,
even more in-depth, ‘individual joints method’,
which allows one to consider the negative and positive
mechanical work produced in different joints of the
same limb. A similar approach was used earlier by Pri-
lutsky et al (1996). It avoids some simplifications of the
previous two methods but retains the general advan-
tage of putting-off the energy recovery mechanisms
that may or may not be employed (Lee et al 2011).
```
### 2. Theory

```
Let us start with the following general principle. For
any instance of the contact phase, a straight line ( y )
through the end-point of a limb can be built, along
which the ground reaction force acts with zero net
mechanical power. In other words, no external work
is done by the limb as a whole if the ground reaction
force is aligned with the ( y ) line (which can more
or less decline from the vertical depending on the
dynamic conditions of the contact phase). In terms
of collisional models (Lee et al 2011), it is the case of
zero collision on the whole-limb level. But, although
the net mechanical power of the limb as a whole is
zero in this case, the power produced in every joint
is not; in some joints it is positive, while in the others
it is negative, so that muscles ( seu actuators) of these
joints work purely against each other (Alexander and
Vernon 1975, Kuznetsov 1995). If the ground reaction
```
(^1) Strictly speaking, some additional torques are produced
by the force of gravity acting upon each limb segment,
and by virtual inertia forces associated with the segments’
accelerations. The forces of gravity are substantially smaller
than the ground reaction force, but the inertia forces can
become considerable at higher speeds of progression. They
reach maxima at limb touch-down and lift-off, just when
the ground reaction force drops to zero. At touch-down,
the inertia forces tend to protract the limb and have to be
balanced by retractor muscles, and at lift-off, the situation is
reversed. However, the inertia forces cannot be optimized as
freely as the ground reaction force direction. All this allows
me to dismiss the limb inertia as well as gravity forces in
order to simplify further considerations.


force is declined (rotated around the point of foot
contact with the ground) to one side of the line ( _y_ ) in
the parasagittal plane of the limb, the external power
produced by the limb as a whole becomes positive,
and if the force declines to the other side, the external
power becomes negative (braking). On the joint level,
this implies prevalence of either positive (in the first
case) or negative (in the second case) joint powers.
Now, let us introduce simplifications. Suppose an
ideal case of weightless limbs and a rigid trunk which
undergoes no pitching, rolling or yawing (if this is a
good idealization or not will be shown by a final com-
parison with the model predictions and the observed
phenomena). Under these simplifications, the line ( _y_ )
is perpendicular to the instantaneous velocity of the
center of mass of the trunk (due to weightless limbs, it
coincides with the common center of mass of the body
as a whole). In particular, in ideal rectilinear progres-
sion over level ground, the velocity of the common
center of mass is kept horizontal and, hence, the line

```
( y ) is vertical (figure 2). If the ground reaction force
declines forward to become more anteriorly directed,
the positive joint powers override the negative ones
and the body is accelerated; if the force declines back-
ward, the balance is reversed and the body is deceler-
ated. In reality, steady locomotion necessitates some
prevalence of positive mechanical work to overcome
air resistance and friction of organs against each
other, as well as to produce non-elastic deformations
of some of the organs or tissues and of the substrate
(footprint production). Often, all these items are rea-
sonably low indeed, and are neglected in the current
model, as in many others before. With all the above
simplifications, the ground reaction is the only force
to be taken into account. The other consequence,
useful for simple modeling, is that under these con-
ditions, in order to sustain the mechanical energy of
the body from stride to stride, positive and negative
work should perfectly cancel each other out—either
immediately, when they are produced simultaneously
```
```
Figure 2. An abstract two-segment limb OAB shown at the end of the contact phase when all the joint coordinates are positive.
(A) Major parameters of the model of zero-work (wheel-like) limb action: the ground reaction force vector is vertical, while the
trunk velocity vector is horizontal. (B) The same with the vector summation of circumferential velocities produced by individual
joints’ movements at the end-point O of the limb. (C) Further development of the model with the horizontal component of the
ground reaction force taken as variable; see explanations in the text.
```

in different joints (accurately speaking, by different
muscles), or somewhat later during the same locomo-
tor cycle (stride).
Positive and negative muscular work has five-fold,
differing metabolic costs (Margaria 1968). However,
to solve a minimum-work problem, one should take
them for the same ‘price’ because, otherwise, the solu-
tion would inevitably come to the complete exclusion
of the more expensive positive work, contradicting the
steady-speed condition: negative work alone would
brake progression. So, I will directly (without any
cost coefficients) sum up the absolute magnitudes of
the mechanical powers produced instantly in all the
joints and regard the minimum value of this sum as
the instantaneous energy-saving optimum. A similar
approach was developed previously for the evalua-
tion of the energy-saving kinematics of a planar three-

```
segment leg in the case of strictly vertical ground reac-
tion force (Kuznetsov 1995). The use of the absolute
magnitudes of mechanical powers is convenient for
geometric considerations of the optimization problem
which I solve. An alternative method, which is better
for an analytic approach, is by joint-by-joint squar-
ing the instantaneous mechanical powers (Abate et al
2016). The third possible method (Ruina et al 2005)
is to consider the positive muscular work alone, based
on the general condition of steady locomotion that the
positive and negative mechanical work of muscles can-
cel each other out over the complete stride. However,
though the last approach was sufficient for overall esti-
mates, it is inappropriate for detailed considerations of
instantaneous optima where the positive and negative
mechanical powers in different joints can be unbal-
anced with each other.
```
```
Figure 3. Joint-fixation principle of the kinematic optimization of a multi-joint limb which develops the ground reaction force
perpendicular to the trunk velocity. (A) Limb OABCKYZ ; optimum performance implies immobilization of all the joints except for
the one ( K ) found perfectly above the end-point O of the limb. (B) Limb OABCYZ ; optimum performance implies immobilization
of all the joints except for the two ( A ) and ( Z ) closest, in angular dimension | θi |, to the ( y ) axis from the opposite sides. (C) Limb
OABC with all the joints at the same side of the ( y ) axis; optimum performance implies immobilization of all the joints except for
the two—the closest ( A ) and the most distant ( B ), in angular dimension | θi |, from the ( y ) axis (this virtually corresponds to the
two-joint limb OAB in figure). (D) Segmental vector contours corresponding to three possible two-joint combinations of the limb
OABC ; see explanations in the text.
```

Let us start with the simplest case of a two-segment
leg _OAB_ (figure 2). Let the vertical ground reaction force
_Fy_ act at right angles to the horizontal trunk velocity _v_ ,
so that there is no mechanical work (collisional work,
according to Lee _et al_ (2011)) on the whole-limb level,
but there are in the limb joints _A_ and _B_. Power devel-
oped in a joint (by its muscles) is the dot product of the
moment vector of the ground reaction force about the
joint and the angular velocity vector _ωi_ of rotation in
this joint, or, that is to say, the dot product of the ground
reaction force vector and the circumferential velocity
vector of the distal limb point _O_ (which develops the
force against the ground) relative to the joint axis. In
our case, the scalar equations for the joint powers are:

```
PA =
```
##### (

```
xA · Fy
```
##### )

(1)·ω _A_ = _Fy_ ·( _xA_ ·ω _A_ ),

```
PB =
```
##### (

```
xB · Fy
```
##### )

(2)·ω _B_ = _Fy_ ·( _xB_ ·ω _B_ ),

where _xA_ and _xB_ are projections of the joint radius vec-
tors _AO_ and _BO_ , respectively, on the line of action of
the external force which, in our simple case, is aligned
with the ( _y_ ) axis. The equations ensure positive power
values for muscle shortening; taken with the negative
sign, they would represent the powers developed by
the external force (the ground reaction) at respective
joints. The geometric sense of these equations is rep-
resented in figure 2(B) by the vector summation of the
circumferential velocities _vA_ and _vB_ of the distal limb
point _O_ produced by angular motions in joints _A_ and
_B_. They can be summed up to give the vector − _v_ of the
posterior shift of the limb end-point _O_ relative to the
trunk; it is horizontal as the vector of the trunk velocity
relative to the ground and has the same absolute mag-
nitude but is in the opposite direction, as indicated by
the negative sign. The roles of the two joints are differ-
ent in the forward progression of the trunk. The joint
where the radius vector is closer to the vertical ensures
the horizontal velocity _v_ , while the second joint com-
pensates for the unwanted collisional ( _sensu_ Lee _et al_
2011) by-product of rotation in the first one. Thus, the
vertical components of the vectors _vA_ and _vB_ represent
the inter-actuator antagonism in the limb:

(3)( _xA_ ·ω _A_ )=−( _xB_ ·ω _B_ ).

Therefore, the net mechanical power of the limb as
a whole is zero ( _PA_ + _PB_ = 0). Then, the amount of the
mechanical antagonism between the joint actuators
can be estimated as:

(4)^2 | _xA_ ·ω _A_ |=^2 | _xB_ ·ω _B_ |=| _PA_ − _PB_ |/ _Fy_ ,

which is represented in figure 2(B) by the vertical
dimension _h_ of the parallelogram formed by the
vectors _vA_ and _vB_. This is what will be minimized in the
next step of the model development.

**2.1. Optimization of the kinematics of the limb
with redundant degrees of freedom**
Consider the same, externally zero-work (non-
collisional _sensu_ Lee _et al_ 2011) situation with the

```
horizontal velocity and vertical ground reaction force
for a multi-segment limb OABC ... K ... YZ (figure
3(A)). In fact, the linkage order of the joints via the
segments does not matter as soon as we consider the
circumferential velocities of the terminal point O
produced by rotations in the individual joints. So,
the principle of the instantaneous equivalence of all
possible linkages can be established. Now, the question
is: which joints should move and which joints should
be fixed (i.e. immobilize the joining segments relative
to each other) in order to minimize the vertical
dimension h of the polygon of the vector summation
of all circumferential velocities, while its horizontal
dimension is still given as − v? The answer is simpler
than the question seems to be.
First of all, when a joint is found on the ( y ) axis
which is the line of action of the vertical ground reac-
tion force, rotation in this joint alone can instantly
imitate a wheel, and all the other joints should be
fixed in order to reduce energy expenses to zero. So,
the multi-segment limb OABCKYZ should best of all
act momentarily like a simple vertical rod OK (figure
3(A)).
When the ( y ) axis does not pass through any
joint (figures 3(B) and (C)), zero expenses cannot
be achieved. Then, the best solution is the use of
those joints only, where mobility ensures the mini-
mum vertical dimension h of the contour of the
vector summation of the respective circumferential
velocities at the limb end-point O. Other joint angles
should be fixed for a while. Now I am going to prove,
by geometric considerations, that the best solution
always involves mobility in two joints only because
the respective parallelogram of circumferential
velocity summations can only be increased vertically
by the addition of any other circumferential velocity
vectors. Remember that the final sum of any vector
contour is given as − v.
The choice of the two joints, where combined
mobility ensures the maximum limb efficiency,
depends on their instantaneous positions relative to
the ( y ) axis. The angular deflections (angles θi ) of the
joint radius-vectors from the vertical are important.
Every joint’s circumferential velocity vector at the limb
end-point O is perpendicular to the respective radius
vector, so the smaller the absolute magnitude | θi | of a
joint angle is, the more horizontal the circumferential
velocity vector is, and the less is its input in the vertical
dimension h of the parallelogram of summation rela-
tive to the horizontal input.
When joints are found on both sides of the ( y ) axis
(limb OABCYZ in figure 3(B)), the best joints to be
used as mobile are those closest to it (by the angle abso-
lute magnitudes | θi |) from the opposite sides—one
posterior and the other anterior to it (in figure 3(B)
these joints are A and Z ). Both the respective circum-
ferential velocity vectors (here vA and vZ ) have a useful
input in the resultant velocity − v of the end-point O. It
is evident that any mobility in all the other joints ( B , C ,
```

_Y_ ) would only increase the energy expenses expressed
by the height _h_ of the vector summation contour.
When all the joints are found on one side of the
( _y_ ) axis (three-segment limb _OABC_ in figure 3(C)),
the one closest to it (by the angle absolute magni-
tude | _θi_ |) should be used again because it can supply
the greatest horizontal component of circumferen-
tial end-point velocity and thus produce the result-
ant − _v_. In our example, it is joint _A_. To compensate
for the harmful (collisional _sensu_ Lee _et al_ 2011) ver-
tical component of its circumferential velocity vec-
tor _vA_ , the best joint is the one most far apart (by the
angle absolute magnitude | _θi_ |) from the ( _y_ ) axis. In
our example, it is joint _B_. Indeed, it supplies more
vertical circumferential velocity vector than joint _C_
(and any other joint that could be placed inside the
angle _AOB_ ). So, _vB_ has the better ratio of its vertical
component (it is useful for compensation against the
motion in joint _A_ ) to horizontal component (which
is harmful in being opposite to the resultant vec-
tor − _v_ ).
Let us generalize the geometric considerations
illustrated in figure 3. Imagine, under the same exter-
nal conditions, a planar serial-linked limb composed
of _n_ joints. _C_ ( _n_ ,2) = _n_ !/2( _n_ − 2)! is the number

```
of two-combinations from the given set of n joints
(for instance, there are ten two-combinations in
figure 3(B), and three in 3(C)). Now, divide the veloc-
ity vector − v of the end-point O relative to the trunk
into C ( n ,2) equal segments, and build on every seg-
ment a two-vector contour for each two-joint com-
bination in the same way as was previously done on
the whole − v vector. This procedure for the limb
shown in figure 3(C) is illustrated in figure 3(D). Every
such segmental contour has its own height where the
magnitude depends on the angles θi of the respective
joint pair. The sum h of all these segmental heights
represents the amount of inter-actuator antagonism
produced in the unique case when motions in all the
joints have proportional input into − v. Generally,
there is only one two-joint combination which has
the minimum height of its segmental vector contour.
Replacing all the others by this best segmental contour
through the whole length of the − v vector, we obtain
the minimum total height h and, in other words, the
minimum amount of antagonism. This replacement
means immobilization (fixation of angles ωi ) of all
the joints except the two. Thus, we have obtained the
final proof of the two-joint preference principle. Their
choice has been already illustrated by figures 3(B) and
```
```
Figure 4. Kinematic algorithm of a generalized parasagittal three-segment Z -like limb, which ensures, at any instant of the contact
phase, the minimum sum of the absolute magnitudes of instantaneous joint powers, provided that the ground reaction force is
perpendicular to the trunk velocity, and so, the limb as a whole acts in zero-work manner—its net mechanical power is zero at any
instant. (A)–(D) Four successive sub-sections of the contact phase characterized by immobilization of different joints. The mobile
joints are indicated by magenta symbols.
```

(C) but let us generalize the rules. The first joint to be
used is the one where the angle from the ( _y_ ) axis has
currently the least absolute magnitude compared to
the rest of the joints: | _θ_ first| is the minimum. The sec-
ond joint to be used is the one which has currently
the greatest absolute magnitude of angular deflection
from the first joint: | _θ_ second − _θ_ first| is the maximum.
This rule is true if the limb joints do not come under
the ground (i.e. below the ( _x_ ) axis through the con-
tact point _O_ ), both when they are found anterior and
posterior to the ( _y_ ) axis (figure 3(B)), and when all of
them are at the same side of the ( _y_ ) axis (figure 3(C)).
By the simple rules of choice of the best joints to
move at any instant, described above, we can easily
deduce the desired kinematic pattern for the whole
contact phase of a three-segment _Z_ -like leg, like that
of a therian mammal, a bird, or the MIT cheetah
robot. Previously, the same algorithm was obtained
by a more elaborate analytic method (Kuznetsov
1995). The most usual kinematic scheme is depicted
in figure 4: (A) when all the joints are posterior to the
point of limb contact with the ground—fix the thigh
(or scapula); (B) after the knee (or shoulder joint) has
passed in front of the point of limb contact with the
ground—fix the ankle (or elbow); (C) after the hip
(or scapular apex) has passed in front of the point of
limb contact with the ground too—fix the knee (or
shoulder joint); (D) after the ankle (or elbow) has
passed in front of the point of limb contact with the
ground too—fix the thigh (or scapula) again, as at the
very beginning.

**2.2. Optimization of the ground reaction force
redundant component**
Having come to the conclusion that the cheapest way
to maintain the instantaneous ground reaction force
as strictly vertical for a planar limb composed of any
number of segments in a chain is to simulate one or the
other two-segment limb by fixation of angles in all the
‘extra’ joints whose motions would otherwise increase
the intra-limb inter-actuator antagonistic expenses,
let us develop the model of the two-segment ideal
limb further and introduce, as a variable, a horizontal
component _Fx_ of the instantaneous ground reaction
force (figure 2(C)). With the given instantaneous
magnitude of _Fy_ , the greater the absolute magnitude
of the variable _Fx_ , the greater the absolute magnitude
of the resultant instantaneous ground reaction
force. In general, a greater external force requires a
greater force from the muscles, including those which
instantly fix ‘extra’ joints. Also, the introduction of the
external force which is parallel to the trunk velocity
_v_ (in the current model it is kept horizontal), results
in the appearance of the external mechanical power
production of the limb as a whole, which changes the
kinetic energy of the trunk—this is the net mechanical
power of the limb as a whole. Can _Fx_ , nevertheless,
help reduce the sum of the absolute magnitudes of
muscular powers required in the two limb joints

```
which are not immobilized according to the algorithm
developed above?
To provide the perfect answer to this question, it
is convenient to check the following hypothesis posed
as a theorem: at any instant of time, the minimum
sum of the absolute magnitudes of mechanical pow-
ers (| PA | + | PB |) produced in the two movable limb
articulations A and B is achieved when the ground
reaction force is aligned with that one which deflects
from the line ( y ) by the least absolute angular abso-
lute magnitude | θi |, provided that the ground reaction
force component Fy acting along the line ( y ) remains
the same. In fact, we have to prove that the required
sum increases if the ground reaction deflects from the
aforementioned articulation (in figure 2(C) it is joint
A ) either backward or forward.
Let us rewrite the equations of joint powers (1) and
(2) with reference to the variable horizontal comp-
onent Fx of the ground reaction force. When it is posi-
tive (forward-directed), its moment about a joint with
positive coordinates ( xi , yi ) is the reverse of that of the
upward vertical component Fy ; that is why the ‘minus’
sign is introduced in the following equations:
(5) PA = Fy ·( xA ·ω A )− Fx ·( yA ·ω A ),
```
```
(6) PB = Fy ·( xB ·ω B )− Fx ·( yB ·ω B ).
Now consider the joint mechanical powers as gen-
eralized linear functions of the variable Fx as if all the
other instantaneous parameters in these equations are
known coefficients. Let us build the general graph for
PA ( Fx ) and PB ( Fx ) (figure 5). First, the net mechanical
power Plimb = Fx · v of the limb as a whole can be drawn
in this graph as a line through the origin of coordi-
nates, which has a positive inclination (dotted line in
figure 5). For a two-joint limb OAB , this line should
represent the simple sum of the graphs of mechani-
cal powers PA ( Fx ) and PB ( Fx ) in its two joints A and B :
Plimb ( Fx ) = PA ( Fx ) + PB ( Fx ). We have just noted that
the joint powers are linear functions of Fx too (equa-
tions (5) and (6)). Let us start building these lines from
their intersections with the abscissa axis Fx. The inter-
section with the abscissa of the PA ( Fx ) graph is ( FAx ,0)
(red circle in figure 5), and that of the PB ( Fx ) graph is
( FBx ,0) (green circle in figure 5), where FxA and FxB are
the values of Fx ensuring that the total ground reaction
force is aligned with joint A or B , correspondingly (see
figure 2(C)). At this Fx value, the mechanical power
in the respective (target) joint falls to zero, while the
mechanical power in the other joint comprises the full
limb power Plimb. In the graph, these joint powers are
obtained by building up vertical segments from the
aforementioned intersection points on the abscissa
axis to the dotted line Plimb ( Fx ). The red segment repre-
sents the mechanical power in joint B when the ground
reaction force is aligned with joint A , and the green
segment represents the mechanical power in joint A
when the ground reaction force is aligned with joint
B. So, in addition to the intersection points with the
```

abscissa axis, we immediately obtain the second points
to build the lines _PA_ ( _Fx_ ) and _PB_ ( _Fx_ ), where the points
have coordinates ( _FBx_ , _PBA_ ) and ( _FxA_ , _PAB_ ), respectively.
Having built these lines (thick line for _PA_ ( _Fx_ ) and dou-
ble line for _PB_ ( _Fx_ ) in figure 5) we obtain their intersec-
tions with the ordinate axis. Automatically, these inter-
section points are just opposite to each other relative to
the origin of coordinates: (0, − _P_^0 ) for the line _PA_ ( _Fx_ )
and (0, _P_^0 ) for the line _PB_ ( _Fx_ ). This is the externally
zero-work situation which was already considered in
the first part of the model (figures 2(A), (B), 3 and 4).
In other words, when the ground reaction force acts
along the line ( _y_ ), no external work is done by the limb,
so some power _P_^0 in one joint (in figure 5 it is the dou-
ble blue segment corresponding to the joint _B_ power)
is entirely cancelled out by respective power − _P_^0 in
the other joint (in figure 5 it is the thick blue segment
corresponding to the joint _A_ power). Hence, the sum
of mechanical power absolute magnitudes in the two
joints is 2| _P_^0 | when _Fx_ is zero in figure 5.

```
Now, using the obtained intersection points with
the abscissa and ordinate axes for coefficients in linear
functions, the general equations of mechanical powers
against Fx for the two joints A and B can be rewritten as
follows:
```
```
(7) PA =− P^0 +( P^0 / FAx )· Fx ,
```
```
(8) PB = P^0 −( P^0 / FxB )· Fx.
```
```
Consider the case when the ground reaction force
is aligned with joint A (variable Fx attains the value FxA ).
Then, all the mechanical power is produced in joint B ,
and its absolute magnitude is:
```
```
| PAB |=| P^0 −( P^0 / FBx )· FxA |=|( 1 − FxA / FBx )· P^0 |.
```
(^) (9)
Now, suppose that the absolute angular deflection
of joint _A_ from the line ( _y_ ) is smaller than that of joint
_B_ , as shown in figure 2:
**Figure 5.** Graph of mechanical power produced in a two-joint limb _OAB_ as a function of the ground reaction force component
_Fx_ which is perpendicular to the line ( _y_ ) of zero net power. The color scheme is the same as in figure 2(C): blue corresponds to the
vertical ground reaction force, red corresponds to the ground reaction force alignment with joint _A_ , green corresponds to the ground
reaction force alignment with joint _B_. Dotted line—the net mechanical power _Plimb_ ( _Fx_ ) of the limb as a whole applied to the trunk;
thick line—mechanical power _PA_ ( _Fx_ ) in joint _A_ which becomes zero when _Fx_ = _FAx_ (red circle); double line—mechanical power
_PB_ ( _Fx_ ) in joint _B_ which becomes zero when _Fx_ = _FBx_ (green circle); grey area—sum | _PA_ ( _Fx_ )| + | _PB_ ( _Fx_ )| of the absolute magnitudes of
the mechanical powers in the two limb joints. In both cases shown, this sum drops to minimum when _Fx_ = _FxA_ , because | _FAx_ |<| _FBx_ |.
(A) Both joints are found anterior to the point _O_ of limb contact with the ground (as in figure 2); (B) Joint _A_ is anterior to point _O_ ,
while joint _B_ is posterior to it.


(10)|θ _A_ |<|θ _B_ |.

```
This means that, as shown in figure 5:
∣∣
FxA
```
##### ∣∣

##### <

##### ∣∣

```
FBx
```
##### ∣∣

##### (11),

```
or
```
(12)−^1 < _FxA_ / _FBx_ <1.

```
From (9) and (12) we get:
∣∣
PAB
```
##### ∣∣

##### < 2

##### ∣∣

##### P^0

##### ∣∣

##### (13).

This means that, when the ground reaction force
is aligned with the limb joint (here it is joint _A_ ) which
is located at the least angle | _θi_ | from the ( _y_ ) axis, the
sum of the mechanical power absolute magnitudes in
the two joints _A_ and _B_ attains the value | _PAB_ | which is
smaller than 2| _P_^0 | representing the value of this sum
when the ground reaction force acts along the line
( _y_ ), its _Fy_ component being the same. The sum con-
sidered shows a monotonous linear increase at either
side of the optimum value _FAx_ —see the grey shading in
figure 5. Simply speaking, the minimum sum occurs
when the steeper one of the two item lines crosses the
abscissa axis, and the steeper is that line which crosses
the abscissa axis closer to the origin of coordinates
(provided that the two cross the ordinate axis strictly
opposite to each other respective to zero). This is what
we wanted to prove: the sum of the mechanical power
absolute magnitudes in the limb joints reaches the
minimum value when the ground reaction force is
aligned with the limb articulation which deflects from
line ( _y_ ) by the least absolute angular absolute mag-
nitude | _θi_ |. Now, the theorem posed above is proved.
Note that the solution found entirely excludes the
inter-actuator antagonism because, with the optimal
ground reaction force, only one actuator in the whole
limb is producing mechanical work at any one time.
In fact, the multi-actuator problem appears to have
a trivial solution: the best action of the limb involves
work production in only one joint at a time, and which
joint is better to actuate depends on the instantaneous
limb configuration. In particular, for the three-seg-
ment _Z_ -like planar limb we can expect that, in steady
level progression, at every moment of the contact
phase, one joint should be fixed (it is chosen according
to the algorithm developed by Kuznetsov (1995) and
generalized here—see figure 4), and the ground reac-
tion force should be aligned with the other joint, being
immediately adjacent in angular dimension | _θi_ | to the
vertical line through point _O_ of the application of the
force of ground reaction to the limb (according to the
current model—figure 6).

### 3. The theory versus real biomechanics

### constrained by muscular actuator

### properties

One can hardly expect that in real animals the joints
can be rigidly fixed in a moment, and the forces can

```
immediately jump from one target point to another.
So, the kinematic and dynamic algorithm developed
above can be, at best, followed rather loosely in
nature. Do mammals really tend to adhere to these
rules? Kuznetsov (1995) has already identified good
agreement with the theory of the forelimb kinematics
in the walking opossum Didelphis virginiana (Jenkins
and Weijs 1979) and of the hind limb kinematics in
the galloping raccoon dog Nyctereutes procyonoides
(Gambaryan 1974). Now, the theoretic predictions can
be tested by comparison with the more comprehensive
data on walking and running kinematics (Fischer
et al 2002) and dynamics (Witte et al 2002) of small
marsupial and placental mammals, possessors of the
typical parasagittal Z -like limbs.
As to limb kinematics, we can expect that, as in fig-
ure 4, the angular velocity in the proximal joint should
drop to zero just at the beginning (touch-down) and
at the end (lift-off ) of the contact phase. This is really
both in the forelimb (as represented by the inclination
of the scapula) and in the hind limb (as represented
by the hip joint angle) (Fischer et al 2002). Also, we
can expect that the angular velocity in the distal joint
should drop to zero before that in the middle joint. In
other words, the two bends of the Z -like zigzag should
not act properly in phase, but the forward-pointing
( M ) bend should mainly flex in the first half of the con-
tact phase, then fix and extend a little just before lift-
off, while the backward-pointing ( D ) bend should flex
a little just after touch-down, then fix, and extend more
in the second half of the contact phase. This asyn-
chrony of motions of the two bends is really observed
in mammalian forelimbs—in the shoulder and the
elbow joints, respectively (Fischer et al 2002). How-
ever, in the hind limb, the flexion–extension motions
in the knee and ankle joints are more synchronized
during the contact phase than those in the shoulder
and elbow than we can expect theoretically. Note that
in Z -like robot limbs, the two bends are usually actu-
ated in perfect synchrony (for example, in the MIT
cheetah robot (Park et al 2014), thus not employing the
redundant degree of freedom in the minimization of
the inter-actuator mechanical antagonism.
As to the limb dynamics, we can expect that, as
in figure 6, the external torques acting upon the limb
joints should approach zero in the following order
(naturally, we do not consider here the zero torques
at the very touch-down and lift-off because they are
caused by the zero values of the ground reaction force
itself, not by its special directionality). In the first half
of the contact phase, the zero should be approached by
the torque in the forward-pointing middle joint, then,
in the middle of the contact phase, by the torque in the
proximal joint, and finally, in the second half of the
contact phase by the torque in the backward-pointing
distal joint. Indeed, this pattern is well pronounced in
the data available for the forelimb of various mam-
mals, but is more obscure in the data on their hind
limbs (Witte et al 2002).
```

Unfortunately, the experimental results on mam-
malian ground reaction forces are very rarely repre-
sented in such a form that allows an explicit considera-
tion of the absence or presence of any target point. The
article by Jayes and Alexander (1978) was a rare excep-
tion. They illustrated the changes of the ground force
angular deflection from the vertical in the hind limb
contact phase of a walking dog as a plot of the tangent
of this angle _θ_ against time. Their figure 11(b) is repro-
duced here by black diamonds in figure 7. The negative
tangent corresponds to the braking horizontal comp-
onent _Fx_ of the ground reaction, typical of the first half

```
of the contact phase, while the positive tangent implies
horizontal acceleration in the second half. The fact that
the graph is almost rectilinear with a gentle inclination
leads to the conclusion of the existence of the target
point above the hip, or shoulder girdle in the case of the
forelimb (Jayes and Alexander 1978).
I have superimposed onto this graph the theor-
etically optimum tangent values (red points in
figure 7). For this purpose, I have analyzed a slow-
motion video sequence of a dog walking over a
treadmill (see supplementary video); it has a low
spatial resolution (i.e. low frame width and height
```
```
Figure 6. Successive stages of the contact phase of a generalized parasagittal three-segment Z -like limb with optimized kinematics
(as in figure 4) and optimized directions of instantaneous ground reaction forces, which ensure, at any instant of the contact phase,
the minimum sum of the absolute magnitudes of instantaneous joint powers. The optimization principle is explained in the text
and is shown in figure 5. (A1) and (A2) The ground reaction force is aligned with the middle joint M, i.e. the knee or the shoulder
joint (A2 corresponds to figure 5(B)); (B1) and (B2) The ground reaction force is aligned with the proximal joint P , i.e. the hip or the
scapular apex (B2 also corresponds to figure 5(B)); (C1) and (C2) The ground reaction force is aligned with the distal joint D , i.e. the
ankle or the elbow (C2 corresponds to figures 2(C) and 5(A)).
```

in pixels and low compression quality) but, what is
more important for force analysis, a high tempo-
ral resolution of 33 frames per hind limb contact
phase). I deduced, for every frame of the right hind
limb contact phase, the energy-saving instantaneous
directions of the ground reaction force (blue lines in
the supplementary video) as predicted by the above
theor etical considerations (figure 6): the ground
reaction vector should follow the hind limb articu-
lations in the order in which they attain the small-
est angular deflections | _θi_ | from the vertical. At first
it follows the knee (circles in figure 7), then the hip
(triangles in figure 7), and finally the ankle (squares
in figure 7). Undoubtedly, due to the general _Z_ -like
leg structure, the three limb joints pass the vertical
line through the point of leg contact with the ground
in the same order in all mammals and birds. So, the
general theoretical graph as a whole looks like a tri-
dent saw. Taken alone, the very segment of the saw is
steeper than the experimental graph (the theor etical
ground reaction force deflects from the vertical more
than the experimental one) because all three limb
articulations are well below Alexander’s point (aster-
isk in figure 1 and in the inset of figure 7). But let us
consider the general inclination of the saw as a whole.
To quantify this inclination, I have calculated the least

```
square linear regression trend of the full set of theor-
etical points and depicted it in the same graph (red
line in figure 7). Except for initial impact disturbance
in the experimental graph (view the second black dia-
mond), the regression line for the theor etical data set
appears to be strikingly close to the real values. This
regression line can be regarded as the theoretical
expectation for the existence of Alexander’s point.
The fact that the experimental regression line runs
a little above the theoretical one may have a very sim-
ple explanation. Probably, the dog of Jayes and Alex-
ander (1978), in this particular stride, was accelerating
forward slightly. If so, the average force of the ground
reaction must have been inclined somewhat forward
from the vertical, and the theoretically predicted
jumps of the instantaneous ground reaction force
from one joint to another should occur a little later
(only about 7% of the contact phase duration, as one
can calculate from the graph in figure 7) than in the
case of the strictly constant velocity of the dog which
we considered theoretically.
The following question arises: why is Alexander’s
point used by the dog, as well as the sheep (Jayes and
Alexander 1978), and, most probably, other therians,
instead of the theoretically more energy-saving saw-
shaped pattern?
```
```
Figure 7. Angular deflections of the ground reaction force from the vertical during the hind limb contact phase of a walking
dog, plotted as the tangent of this angle θ against time which is measured as the percentage of the contact phase duration. Black
diamonds represent the real values (from Jayes and Alexander (1978)). The red points represent the theoretical prediction based
on the kinematics of the dog walking over the treadmill: circles—force is aligned with the knee, triangles—force is aligned with the
hip, squares—force is aligned with the ankle. Straight lines and equations in respective colors represent the least square regressions
of the two data sets. The video frame inset shows the posture where the hip approaches the same negative angular deflection from
the vertical as the positive one of the knee ( θP = − θM ), and the first shift of the ground reaction force should theoretically occur.
Alexander’s target point for the hind limb is shown by an asterisk (according to figure 1).
```

The reason may be in the insufficient rapidity of
limb muscles to perform abrupt changes of the ground
reaction force. A similar explanation was suggested by
Jayes and Alexander when they determined that the
chelonian walking gait is not as stable as it could be with
the more abrupt changes of the ground reaction force
magnitude (Jayes and Alexander 1980). Obviously,
mammalian muscles are faster than chelonian ones,
but the saw-shaped deflection pattern of the ground
reaction force, which seems to be theoretically desir-
able in mammals, is much more abrupt than what was
supposed as the best solution for chelonians. In chelo-
nians, only the magnitude of the vertical component
Fy of the ground reaction force was to be changed step-
wise if their muscles were faster (Jayes and Alexander
1980), not the direction of its horizontal component
Fx, which, in the saw-shaped model, has to be instantly
switched from positive to negative twice during the
contact phase (both switches are shown in figure 6, and
the first switch is depicted in the inset of figure 7). Even
mammalian muscles cannot switch on and off imme-
diately, which would be the necessary ability to change
the target point of the ground reaction force from the
knee to the hip and finally to the ankle joint. Briefly
speaking, Alexander’s point exists for two reasons:
(1) the tendency to minimize, at any instance of the
contact phase, the sum of mechanical power absolute
magnitudes in all the limb joints, and (2) the inability
of mammalian musculature to contract fast enough to
perfectly fulfill the first condition. The penalty for this
muscular sluggishness is the additional work of the
muscles of different joints against each other, inevita-
bly associated with keeping the ground reaction force
in line with Alexander’s point, that is closer to the ver-
tical than the optimal target joints. However, this mus-
cular ‘antagonism’ can be reduced with the help of
muscles crossing the two moving joints (Elftman 1939,
1940, Kuznetsov 1995, Prilutsky _et al_ 1996, Junius _et al_
2017). In robotics, the role of such muscles in reducing
mechanical antagonism can be effectively performed
by redundantly actuated parallel mechanisms added to
serial-linked limbs (Lee _et al_ 2017).
Finally, it should be noted that the discrepancy
between the theoretic prediction and the actual
ground reaction force behavior may be subject to scal-
ing effects. Larger animals have lower frequencies of
leg movements, and hence, their muscles possess some
advantage, as compared to smaller animals, to switch
on and off in proper time to ensure the presumed saw-
shaped pattern of the foot interaction with the ground.
Indeed, some traces of the expected saw-shaped pat-
tern can be noticed among recordings of equine
ground reaction forces (Niki _et al_ 1984). The presented
graphs of the horizontal ground reaction force comp-
onent of both forelimb and hind limb (figure 2 of the
paper cited) not only show the typical sign change
(from braking to propulsive) near the middle of the
contact phase, but in addition, there are two drops
in the magnitude of the horizontal component. The

```
first one occurs in the first half of the contact phase,
between the typical initial impact disturbance and the
sign change, just near the point where the ground reac-
tion force is expected to jump from the first target joint
(knee or shoulder) to the second one (hip or scapular
pivot). The second drop occurs after the sign change,
just near the point where the ground reaction force is
expected to jump from the second target joint (hip or
scapular pivot) to the third one (ankle or elbow).
```
### 4. Conclusions and future work

```
The theoretic model developed above suggests
simple rules for the minimization of the mechanical
antagonism between actuators in any serial-linked
planar walking limb. The minimization is achieved
through the optimal employment of the horizontal
component of the ground reaction force and of the
redundant degrees of freedom. These rules can be
immediately applied in robotics both in the design
of new limb architectures and in the improvement of
kinematics and force control of the limbs of existing
devices, such as the MIT cheetah robot. This will
help to create walking machines with a lower cost of
transport than that of animals. Artificial actuators
may be better to strictly follow the optimization rules
than natural muscles. So, for robots, the replacement
of theoretically optimal target points for the ground
reaction force with a single averaged Alexander’s point
may be unnecessary.
The approach which I have introduced above
to approximate directions of the ground reaction
force acting on the hind limb of a walking dog can be
developed in a new method of biomechanical analy-
sis. Frequently, especially for rare animals, the video
recordings of locomotion are unique, and force plate
recordings are impossible to obtain. Before, such video
recordings were regarded as useless for the analysis of
limb mechanics. Now, the directions of ground reac-
tion forces can be restored from a kinematical analysis
of video recordings alone, conducted in three steps.
First, the optimal lines of action of ground reaction
forces should be drawn on every video frame, as is done
in the supplementary video (the principle is depicted in
figure 6). Second, the tangents of declination of these
lines from the vertical should be plotted against time
and approximated by a least-square linear regression
trend, as in figure 7. Third, the regression line should
be converted back into angles from the vertical, and
respective improved directions of the ground reaction
forces should be drawn on the video frames. Potentially,
this method can be appropriate even for the analy-
sis of poor-quality home video. However, the method
requires testing on a more representative variety of
mammalian and avian legs and gaits in order to show
that the perfect result of my analysis was not occasional.
Further research of the mechanical antagonism
problem from both sides, biomechanics and robot-
ics, will be helpful reciprocally for an understanding of
```

the energetic fundamentals of animal locomotion for
biomimetic applications. One of the major tasks which
could be solved next is to test the hypothesis arising
from my research that, in the family of serial-linked
planar limbs, the three-segment _Z_ -like structure found
in mammals and birds and copied by the MIT cheetah
robot is the best in the minimization of the inter-actu-
ator mechanical antagonism. Recently, it was shown
by modeling that three-segment limbs are more eco-
nomic than two-segment ones in jumping and walking
over irregular ground (Ha _et al_ 2016). I propose that
the energetic advantage of the three-segment _Z_ -like
limb has a more general nature. My preliminary idea
is that the _Z_ -like structure supplies the widest possi-
ble sector between the available target points (joints)
located anterior and posterior to the limb axis ( _OP_ ),
as compared to any serial-linked systems composed
either of two segments or of more than three segments,
or of three segments arranged so that both bends of the
limb point to the same side.

### Acknowledgments

I would like to thank the anonymous reviewers and
Boris I Prilutsky for their suggestions, which have
greatly improved this manuscript.
The slow-motion video of a dog walking over the
treadmill (see supplementary material) was kindly
supplied by Maris Multimedia Ltd and its licensors. It
is from the ‘How Animals Move’ CD produced in 1995
under the authorship of R McNeill Alexander and edi-
torship of myself (Alexander 1995).
The work was supported by the Russian Foun-
dation for Basic Research (grants 14-04-01132 and
17-04-00954).

### ORCID iDs

Alexander N Kuznetsov https://orcid.org/0000-
0002-9928-

### References

Abate A, Hurst J W and Hatton R L 2016 Mechanical antagonism in
legged robots _Robot.: Sci and Syst. XII_
Agarwal S, Mahapatra A and Roy S S 2012 Dynamics and optimal
feet force distributions of a realistic four-legged robot _Int. J.
Robot. Autom._ **1** 223 – 34
Alexander R McN 1976 Mechanics of bipedal locomotion
_Perspectives in Experimental Biology 1: Zoology. Proceedings of
the 50th Anniversary Meeting of the Society for Experimental
Biology (1974: Cambridge University)_ ed P S Davies _et al_
(Oxford: Pergamon) pp 493– 504
Alexander R McN 1977 Mechanics and scaling in terrestrial
locomotion _Scale Effects in Animal Locomotion. International
Symposium on Scale Effects in Animal Locomotion (1975:
Cambridge University)_ ed T J Padley (London: Academic) pp
93 – 110
Alexander R McN 1980 Optimum walking techniques for
quadrupeds and bipeds _J. Zool._ **192** 97 – 117
Alexander R McN 1991 Energy-saving mechanisms in walking and
running _J. Exp. Biol._ **160** 55 – 69

```
Alexander R McN 1995 How animals move The Discovery Channel
and Maris Multimedia ed A Kuznetsov (CD-ROM) (London:
Maris Multimedia)
Alexander R McN and Jayes A S 1980 Fourier analysis of forces
exerted in walking and running J. Biomech. 13 383 – 90
Alexander R McN and Vernon A 1975 The mechanics of hopping by
kangaroos (Macropodidae) J. Zool. 177 265 – 303
Biewener A A 1983 Allometry of quadrupedal locomotion: the
scaling of duty factor, bone curvature and limb orientation to
body size J. Exp. Biol. 105 147 – 71
Biewener A A 1989 Scaling body support in mammals: limb posture
and muscle mechanics Science 245 45 – 8
Blickhan R, Andrada E, Müller R, Rode C and Ogihara N 2015
Positioning the hip with respect to the COM: consequences for
leg operation J. Theor. Biol. 382 187 – 97
Cahill N M, Sugar T, Holgate M and Schroeder K 2017
Understanding power loss due to mechanical antagonism and
a new power-optimal pseudoinverse for redundant actuators
Proc. of the ASME 2017 Int. Design Engineering Technical
Conf. and Computers and Information in Engineering Conf. p
V05BT08A
Donelan J M, Kram R and Kuo A D 2002 Simultaneous positive
and negative external mechanical work in human walking J.
Biomech. 35 117 – 24
Elftman H 1939 The function of muscles in locomotion Am. J.
Physiol. 125 357 – 66
Elftman H 1940 Work done by muscles in running Am. J. Physiol.
129 672 – 84
Fischer M S 1998 Die Lokomotion von Procavia capensis (Mammalia:
Hyracoidea): zur evolution des bewegungssystems bei
saügetieren Abh. Naturw. Ver. Hamburg 33 1 – 188
Fischer M S, Schilling N, Schmidt M, Haarhaus D and Witte H 2002
Basic limb kinematics of small therian mammals J. Exp. Biol.
205 1315 – 38
Gambaryan P P 1974 How Mammals Run. Anatomical Adaptations
(New York: Wiley)
Günther M, Keppler V, Seyfarth A and Blickhan R 2004 Human leg
design: optimal axial alignment under constraints J. Math.
Biol. 48 623 – 46
Ha S, Coros S, Alspach A, Kim J and Yamane K 2016 Task-based limb
optimization for legged robots 2016 IEEE/RSJ Int. Conf. on
Intelligent Robots and Systems (IROS) pp 2062– 8
Jayes A S and Alexander R McN 1978 Mechanics of locomotion
of dogs ( Canis familiaris ) and sheep ( Ovis aries ) J. Zool.
185 289 – 308
Jayes A S and Alexander R McN 1980 The gaits of chelonians:
walking techniques for very low speeds J. Zool. 191 353 – 78
Jenkins F A and Weijs W A 1979 The functional anatomy of the
shoulder in the Virginia opossum ( Didelphis virginiana ) J.
Zool. 188 379 – 410
Junius K, Moltedo M, Cherelle P, Rodriguez-Guerrero C,
Vanderborght B and Lefeber D 2017 Biarticular elements as
a contributor to energy efficiency: biomechanical review and
application in bio-inspired robotics Bioinspir. Biomim.
12 061001
Kar D C, Kurien Issac K and Jayarajan K 2001 Minimum energy
force distribution for a walking robot J. Robot. Syst.
18 47 – 54
Kuznetsov A N 1985 Comparative functional analysis of the fore
and hind limbs in mammals Zool. Zh. 64 1862 – 7 (in Russian)
Kuznetsov A N 1995 Energetical profit of the third segment in
parasagittal legs J. Theor. Biol. 172 95 – 105
Lee D V, Bertram J E, Anttonen J T, Ros I G, Harris S L and
Biewener A A 2011 A collisional perspective on quadrupedal
gait dynamics J. R. Soc. Interface 8 1480 – 6
Lee J, Lee G and Oh Y 2017 Energy-efficient robotic leg design using
redundantly actuated parallel mechanism IEEE Int. Conf. on
Advanced Intelligent Mechatronics (AIM) pp 1203– 8
Margaria R 1968 Positive and negative work performances and
their efficiencies in human locomotion Int. Z. Angew. Physiol.
25 339 – 51
Maus H-M, Lipfert S W, Gross M, Rummel J and Seyfarth A 2010
Upright human gait did not provide a major mechanical
challenge for our ancestors Nat. Commun. 1 70
```

Niki Y, Ueda Y and Masumitsu H 1984 A force plate study in equine
biomechanics. 3. The vertical and fore-aft components of floor
reaction forces and motion of equine limbs at canter _Bull.
Equine Res. Inst._ **21** 8 – 18
Park H W, Chuah M Y and Kim S 2014 Quadruped bounding
control with variable duty cycle via vertical impulse scaling
_2014 IEEE/RSJ Int. Conf. on Intelligent Robots and Systems
(IROS 2014)_ pp 3245– 52
Prilutsky B I, Petrova L N and Raitsin L M 1996 Comparison of
mechanical energy expenditure of joint moments and muscle
forces during human locomotion _J. Biomech._ **29** 405 – 15
Ruina A, Bertram J E and Srinivasan M 2005 A collisional model
of the energetic cost of support work qualitatively explains
leg sequencing in walking and galloping, pseudo-elastic leg

```
behavior in running and the walk-to-run transition J. Theor.ё
Biol. 237 170 – 92 
Seok S, Wang A, Chuah M Y, Otten D, Lang J and Kim S 2013 Design
principles for highly efficient quadrupeds and implementation
on the MIT Cheetah robot 2013 IEEE Int. Conf. on Robotics and
Automation (ICRA) pp 3307– 12
Waldron K J and Kinzel J L 1981 The relationship between actuator
geometry and mechanical efficiency in robots Proc. of the
4th Symp. on Theory and Practice of Robots and Manipulators
pp 366– 74
Witte H, Biltzinger J, Hackert R, Schilling N, Schmidt M, Reich C
and Fischer M S 2002 Torque patterns of the limbs of small
therian mammals during locomotion on flat ground J. Exp.
Biol. 205 1339 – 53
```

