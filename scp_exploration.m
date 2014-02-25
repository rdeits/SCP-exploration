draw_pause = 0.05;
d_check = 0.1;
d_safe = 0.05;

r = RigidBodyManipulator();

n_keyframes = 10;
n_obs = 1;

for i=1:n_obs
  r = addRobotFromURDF(r,'LargeBrick.urdf',zeros(3,1),zeros(3,1),struct('floating',true));
end

for i=1:n_keyframes
  r = addRobotFromURDF(r,'SmallBrick.urdf',zeros(3,1),zeros(3,1),struct('floating',true));
end
  
obs_ndx = 1 + (1:n_obs);
key_ndx = 1 + n_obs + (1:n_keyframes);

lcmgl = drake.util.BotLCMGLClient(lcm.lcm.LCM.getSingleton(),'bullet_collision_closest_points_test');


v = r.constructVisualizer(struct('viewer', 'RigidBodyWRLVisualizer'));
% v = r.constructVisualizer();

q = zeros(getNumDOF(r),1);

p0 = [-2;0;0;0;0;0];
pf = [2;0;0;0;0;0];

lambda = linspace(0,1,n_keyframes);

for i = 1:length(key_ndx)
  k = key_ndx(i);
  q_ndx = (k-2)*6+(1:6);
  p = p0 + (pf-p0).*lambda(i);
  q(q_ndx) = p;
end

kinsol = doKinematics(r,q);
pts = contactPositions(r,kinsol);
tol = 1e-6;

% We'll linearize the relevant contacts as sd = sd0 + dsd' * (q - q0)
% So our constraint is g(x) = d_safe - (sd0 + dsd' * (q - q0)) <= t
% -dsd' * q - t <= - d_safe + sd0 - dsd' * q0
sd0 = zeros(n_obs * n_keyframes,1);
dsd = zeros(size(sd0,1), size(q,1));
sd_ndx = 1;


for k = key_ndx
  
  [ptA,ptB,normal,distance,JA,JB] = pairwiseClosestPoints(r,kinsol,k,obs_ndx(1));
  
  if distance <= d_check
    dsd(sd_ndx,:) = (normal' * JA);
    sd0(sd_ndx) = distance;
    sd_ndx = sd_ndx + 1;
  end

  v.draw(0,[q;0*q]);

  lcmgl.glColor3f(1,0,0); % red
  lcmgl.sphere(ptA,.05,20,20);

  lcmgl.glColor3f(0,1,0); % green
  lcmgl.sphere(ptB,.05,20,20);

  lcmgl.glColor3f(0,0,0); % black
  lcmgl.text((ptB+ptA)/2+0.5*[0;0;1],num2str(distance),0,0);
  signed_dist = (ptA - ptB)' * normal
  valuecheck(distance, signed_dist, 1e-6);

  lcmgl.glColor3f(.7,.7,.7); % gray

  lcmgl.switchBuffers();

%   pause(0.25);
end

sd0(sd_ndx+1:end) = [];
dsd(sd_ndx+1:end,:) = [];

n_slack = length(sd0);
nv = length(q) + n_slack;
Q = diag(ones(nv,1));
c = [zeros(length(q),1); ones(n_slack,1)];
max_slack = 1000;
lb = [-inf(length(q),1); zeros(n_slack,1)];
ub = inf(nv,1);
lb(1:12) = q(1:12);
lb(3:6:length(q)) = 0;
lb(length(q)-5:length(q)) = q(end-5:end);
ub(1:12) = q(1:12);
ub(3:6:length(q)) = 0;
ub(length(q)-5:length(q)) = q(end-5:end);


Asd = -dsd;
Asd = [Asd, diag(-ones(n_slack,1))];
A_rel = zeros((n_keyframes-1) * 3, size(Asd,2));
for k = 1:3
  for key0 = 1:(n_keyframes-1)
    A_rel((key0-1)*3+k,key0*6+k) = -1;
    A_rel((key0-1)*3+k,(key0+1)*6+k) = 1;
  end
end
b_rel = 1 * ones(size(A_rel,1),1);

bsd = sd0 - (d_safe + zeros(size(sd0))) - dsd * q;

clear model params
model.A = sparse([Asd; A_rel]);
model.rhs = [bsd;b_rel];
model.obj = c;
model.sense = '<';
model.lb = lb;
model.ub = ub;
model.Q = sparse(Q);

result = gurobi(model);
qstar = result.x(1:length(q));
reshape(qstar,6,[])
v.draw(0,[qstar;0*qstar]);