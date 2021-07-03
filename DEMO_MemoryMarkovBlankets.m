function DEMO_MemoryMarkovBlankets
% This script reproduces the figures presented in the paper 'Memory and
% Markov blankets' (Parr, Da Costa, Heins, Ramstead, and Friston), written
% for the Entropy (MDPI) special issue 'Entropies, Information Geometry and 
% Fluctuations in Non-equilibrium Systems.' The central idea of these
% demonstrations is that, on taking a system whose steady state density
% factorises according to the conditional dependency structure of a Markov
% blanket, imposition of new initial conditions on the blanket states
% results in a transient conditional dependency across the blanket. This
% means it is possible for internal states to maintain some memory of
% external states, even in the absence of informative blanket states, until
% steady state has been restored.

% Preliminaries
%==========================================================================
rng default % For reproducibility
set(0,'DefaultAxesFontName','Times New Roman','DefaultTextFontName','Times New Roman');

T  = 16;
dt = 1/32;

close all

% Gaussian non-equilibrium steady state (NESS)
%==========================================================================
% We start by stipulating a steady state density that preserves the
% conditional dependency structure of a Markov blanket. Specifically, for
% external (h), internal (m), and blanket (b) states, we have a
% multivariate normal distribution that can be factorised as:
%
%     p(h,m,b) = p(b)p(m|b)p(h|b)
%
% This is constructed by assuming zero covariance between h and m. We then
% define the matrices responsible for solenoidal and dissipative components
% of the stochastic differential equation that leads to this NESS. Using
% the Laplace assumption, we then simulate the evolution of the probability
% density following a perturbation (here, an observation of a blanket
% state). Our aim is to establish the influence of the steady state
% covariance, the dissipative, and the solenoidal contributions to the flow
% on the time taken for the covariance between external and internal states
% to return to zero following perturbation.

% NESS expectation and covariance
%--------------------------------------------------------------------------
pE = zeros(3,1);   % Expected internal, blanket, and external state
pC = inv([2 1 0;   % Covariance of internal states,
          1 2 1;   % ... blanket states,
          0 1 2]); % ... and external states

% Solenoidal and dissipative flows
%--------------------------------------------------------------------------
Q = [0 -1  0;      % Solenoidal  (antisymmetric)
     1  0 -1;
     0  1  0];

R = eye(3)/4;      % Dissipative (symmetric)

% Initialise expectation and covariance
%--------------------------------------------------------------------------
Ep  = ones(3,1);
Cp  = eye(3)/4;

% Compile into structure and integrate Fokker-Planck Equation
%--------------------------------------------------------------------------
D.Q  = Q;
D.R  = R;
D.pE = pE;
D.pC = pC;
D.Ep = Ep;
D.Cp = Cp;
D.T  = T;
D.dt = dt;
D.x  = Ep;

D = mmb_FPE_Laplace(D);

% Integrate stochastic differential equation multiple times
%--------------------------------------------------------------------------
N = 512;                                     % Number of instances
X = zeros(size(D.Ep,1),size(D.Ep,2),N);      % Initialise
for i = 1:N
    d = mmb_SDE(D);
    X(:,:,i) = d.x;
end

% Plot stochastic simulation
%--------------------------------------------------------------------------
% The upper left plot shows many instantiations of the system plotted in
% 3D. The upper right and middle row of plots show this plotted from 3
% different orthogonal views. The lower plot shows the average behaviour
% over time, illustrating convergence to the steady state via a series of
% damped oscillations.

f1 = figure(1); clf
mmb_SDE_plot(f1,X,dt)
clear d

% Plot density dynamics under the Laplace assumption
%--------------------------------------------------------------------------
% This figure plots the same system as above, but integrates the
% Fokker-Planck equation under Gaussian assumptions, as opposed to the SDE. 

f2 = figure(2); clf
mmb_Laplace_plot(f2,D)

% Simulate a system conditioned upon a given initial blanket state
%--------------------------------------------------------------------------
% As our focus is on a time-dependent system given an initial blanket
% state, we initialise with a density consistent with steady state in all
% but the precision of the blanket states, which are set to be very large,
% as if making an observation.

D = rmfield(D,'x');
D.Ep      = pE;
P         = inv(pC);
P(2,2)    = exp(4);
D.Cp      = inv(P);
D.T       = 4;

% Measures of memory span
%--------------------------------------------------------------------------
% This plot illustrates several different ways in which we could measure
% the memory span of the stochastic system specified above. These include
% the precision corresponding to dependencies between internal and
% external, the mutual information between these states, conditioned upon
% the blanket, the information length of the path travelled from initial
% conditions to the steady state, and the reciprocal of the real parts of
% the eigenvalues of the Jacobian, which can be interpreted as time
% constants for the decay of each linear mixture of covariance components.

f3 = figure(3);
mmb_memory_plot(f3,D)

% Influence of each variable on memory span
%--------------------------------------------------------------------------
D.Ep      = pE;
D.Cp      = inv(P);
D.T       = 8;

% Rate of dissipation
%--------------------------------------------------------------------------
for i = 1:8
    d(i)   = D;
    d(i).R(1,1) = D.R(1,1)*i/4;
    d(i).R(3,3) = D.R(3,3)*i/4;
end

f4 = figure(4);
mmb_memory_plot(f4,d)
f4.Name = 'Memory and dissipative flow';

% Rate of solenoidal flow
%--------------------------------------------------------------------------
for i = 1:8
    d(i)   = D;
    d(i).Q = D.Q*i/4;
end

f5 = figure(5);
mmb_memory_plot(f5,d)
f5.Name = 'Memory and solenoidal flow';

% Variance of NESS
%--------------------------------------------------------------------------
for i = 1:8
    d(i)   = D;
    H       = inv(D.pC);
    H(2,2)  = H(2,2)*i/4;
    H(1,2)  = H(1,2)*sqrt(i/4);
    H(2,1)  = H(2,1)*sqrt(i/4);
    H(2,3)  = H(2,3)*sqrt(i/4);
    H(3,2)  = H(3,2)*sqrt(i/4);
    H       = H*4/i;
    d(i).pC = inv(H);
    P       = inv(d(i).pC);
    P(2,2)  = exp(4);
    d(i).Cp = inv(P);
end

f6 = figure(6);
mmb_memory_plot(f6,d)
f6.Name = 'Memory and NESS covariance';
clear D d

% non-Gaussian non-equilibrium steady state (NESS)
%==========================================================================
% In the above, we assumed the NESS was in the family of multivariate
% Gaussian densities. This let us examine three different variables and
% their role in the length of time for which a memory is retained. However,
% this leaves a fourth factor unexamined. This is the form of the NESS
% density. Here, we assess the same dynamics for a categorical
% distribution.

T = 8;

% Categorical NESS
%--------------------------------------------------------------------------
Sb(1,1,:) = [2,6,2]/10; Sb = repmat(Sb,[3 3 1]);               % Blanket states
Sh(:,1,:) = [2 6 2;6 2 2;2 2 6]/10; Sh = repmat(Sh,[1 3 1]);   % External given blanket states
Sm(1,:,:) = [2 6 2;6 2 2;2 2 6]/10; Sm = repmat(Sm,[3 1 1]);   % Internal given blanket states

% Dissipative and solenoidal flows
%--------------------------------------------------------------------------
R = -ones(length(Sb(:)))/128;  % Dissipative flows
R = R + diag(diag(R)); 
R = R - diag(sum(R));         

Q = [0 -1 1;1 0 -1;-1 1 0];    % Solenoidal flows
Q = kron(Q,kron(Q,Q));
Q = Q/512;

% Initialise
%--------------------------------------------------------------------------
sh = Sh;
sm = Sm;
sb = zeros(1,1,3); sb(2) = 1; sb = repmat(sb,[3 3 1]);

% Compile and solve
%--------------------------------------------------------------------------
D.Sb = Sb;
D.Sh = Sh;
D.Sm = Sm;
D.sb = sb;
D.sh = sh;
D.sm = sm;
D.R  = R;
D.Q  = Q;
D.T  = T; 
D.dt = dt;

D.T = 2;
D = mmb_Cat_ME(D); % Numerically integrate
D.T = T;

% Plotting
%--------------------------------------------------------------------------
f7 = figure(7);
mmb_Cat_plot(f7,D) 
D = rmfield(D,'s');

f8 = figure(8);
mmb_Cat_memory_plot(f8,D);

% Rate of dissipation
%--------------------------------------------------------------------------
for i = 1:8
    d(i)   = D;
    d(i).R = D.R*i/4;
end

f9 = figure(9);
mmb_Cat_memory_plot(f9,d)
f9.Name = 'Memory and dissipative flow';

% Rate of solenoidal flow
%--------------------------------------------------------------------------
for i = 1:8
    d(i)   = D;
    d(i).Q = D.Q*i/4;
end

f10 = figure(10);
mmb_Cat_memory_plot(f10,d)
f10.Name = 'Memory and solenoidal flow';

% Precision of NESS
%--------------------------------------------------------------------------
for i = 1:8
    d(i)    = D;
    d(i).Sh = D.Sh.^(i/4)./repmat(sum(D.Sh.^(i/4),1),[3 1 1]);
    d(i).sh = d(i).Sh;
    d(i).Sm = D.Sm.^(i/4)./repmat(sum(D.Sm.^(i/4),2),[1 3 1]);
    d(i).sm = d(i).Sm;
    d(i).Sb = D.Sb.^(i/4)./repmat(sum(D.Sb.^(i/4),3),[1 1 3]);
end

f11 = figure(11);
mmb_Cat_memory_plot(f11,d)
f11.Name = 'Memory and NESS precision';

% FUNCTIONS FOR CONTINUOUS DENSITIES
%==========================================================================

function D = mmb_FPE_Laplace(d)
% This function takes a structure, d, comprising the following fields:
% Q     - Solenoidal flow matrix
% R     - Dissipitive flow matrix
% pE    - Expectation at NESS
% pC    - Covariance at NESS
% Ep    - Initial expectation
% Cp    - Initial covariance
% T     - Duration over which to integrate
% dt    - Integration step
% and numerically integrates the associated Fokker-Planck equation under a
% Laplace assumption, returning a structure D whose Ep and Cp fields
% include the values of the expectation and covariance at each iteration.
%__________________________________________________________________________

% Initialise
%--------------------------------------------------------------------------
D  = d;
T  = d.T;                                    % Integration duration
dt = d.dt;                                   % Integration step
R  = d.R;                                    % Dissipative flow
Q  = d.Q;                                    % Solenoidal flow
E  = d.Ep;                                   % Initial expectation
C  = d.Cp;                                   % Initial covariance
Je = -(R-Q)/d.pC;                            % Jacobian for expectation
Jc = -(kron((R-Q)/d.pC,eye(size(C,1)))...    % Jacobian for covariance
    + kron(eye(size(C,1)),(R-Q)/d.pC));

% Numerically integrate
%--------------------------------------------------------------------------
for i = 1:round(T/dt)
    E             = E(:) + Je\(expm(Je*dt)-eye(size(Je,1)))*(Je*(E - d.pE));
    C(:)          = C(:) + Jc\(expm(Jc*dt)-eye(size(Jc,1)))*(2*R(:) + Jc*C(:));
    D.Ep(:,i+1)   = E;
    D.Cp(:,:,i+1) = C;
end

% Save Jacobians
%--------------------------------------------------------------------------
D.Je = Je;
D.Jc = Jc;

function D = mmb_SDE(d)
% This function takes a structure, d, comprising the following fields:
% Q     - Solenoidal flow matrix
% R     - Dissipitive flow matrix
% pE    - Expectation at NESS
% pC    - Covariance at NESS
% x     - Initial conditions
% T     - Duration over which to integrate
% dt    - Integration step
% and numerically integrates the associated stochastic differential
% equation, returning a structure D whose x field includes the value of x
% for every iteration.
%__________________________________________________________________________

% Initialise
%--------------------------------------------------------------------------
D  = d;
T  = d.T;                                    % Integration duration
dt = d.dt;                                   % Integration step
x  = d.x;
R  = d.R;
Q  = d.Q;
J  = -(R-Q)/d.pC;

% Numerically integrate
%--------------------------------------------------------------------------
for i = 1:round(T/dt)
    dw       = (mvnrnd(zeros(size(x,1),1),2*R*dt))';
    x        = x + J\(expm(J*dt)-eye(size(J,1)))*(J*(x - d.pE)) + dw;
    D.x(:,i+1) = x;
end

function mmb_SDE_plot(f,X,dt)
% Plotting for the stochastic simulation
%__________________________________________________________________________

% Figure properties
%--------------------------------------------------------------------------
f.Name = 'Stochastic dynamics';
f.Color = 'w';
f.Position = [500 50 500 550];

% Plot average trajectories
%--------------------------------------------------------------------------
subplot(3,2,5:6)
plot((0:size(X,2)-1)*dt,mean(X,3)')
xlim([0 size(X,2)*dt])
xlabel('Time')
title('Average trajectories')
legend('Internal','Blanket','External')
box off

% Plotting (animation)
%--------------------------------------------------------------------------
for i = 1:size(X,2)
    subplot(3,2,1)
    plot3(squeeze(X(1,i,:)),squeeze(X(2,i,:)),squeeze(X(3,i,:)),'.k')
    axis([-4 4 -4 4 -4 4])
    axis square
    xlabel('Internal')
    ylabel('Blanket')
    zlabel('External')
    title('Markov blanket')
    hold off
    
    subplot(3,2,2)
    plot(squeeze(X(1,i,:)),squeeze(X(2,i,:)),'.k')
    axis([-4 4 -4 4])
    axis square
    xlabel('Internal')
    ylabel('Blanket')
    title('Internal and blanket')
    box off
    hold off
    
    subplot(3,2,3)
    plot(squeeze(X(3,i,:)),squeeze(X(2,i,:)),'.k')
    axis([-4 4 -4 4])
    axis square
    xlabel('External')
    ylabel('Blanket')
    title('External and blanket')
    box off
    hold off
    
    subplot(3,2,4)
    plot(squeeze(X(1,i,:)),squeeze(X(3,i,:)),'.k')
    axis([-4 4 -4 4])
    axis square
    xlabel('Internal')
    ylabel('External')
    title('Internal and external')
    box off
    hold off
    drawnow
end

function mmb_Laplace_plot(f,D)
% Plotting for the density dynamics simulation
%__________________________________________________________________________

% Figure properties
%--------------------------------------------------------------------------
f.Name = 'Density dynamics under the Laplace assumption';
f.Color = 'w';
f.Position = [500 50 500 550];

% Preliminaries
%--------------------------------------------------------------------------
E     = D.Ep;                                   
C     = D.Cp;                    
[X,Y] = meshgrid(-4:1/8:4,-4:1/8:4);

% Plot mode
%--------------------------------------------------------------------------
subplot(3,1,2)
plot((0:size(E,2)-1)*D.dt,E')
xlim([0 size(E,2)*D.dt])
xlabel('Time')
title('Trajectory of mode')
legend('Internal','Blanket','External')
box off

% Plot precision
%--------------------------------------------------------------------------
P = zeros(size(C));
for i = 1:size(C,3)
    P(:,:,i) = inv(C(:,:,i));
end
c = reshape(P,[size(C,1)*size(C,2),size(C,3)]);
c = c([1 2 3 5 6 9],:);

subplot(3,1,3)
plot((0:size(E,2)-1)*D.dt,c')
xlim([0 size(E,2)*D.dt])
xlabel('Time')
title('Trajectory of precision')
legend('Internal-internal','Internal-blanket','Internal-external','Blanket-blanket','Blanket-external','External-external')
box off

% Plotting (animation)
%--------------------------------------------------------------------------
for i = 1:size(E,2)
    subplot(3,3,1)
    f  = @(x,y) exp(-(([x y] - E(1:2,i)')*inv(C(1:2,1:2,i))*([x; y] - E(1:2,i)))/2)/(2*pi*sqrt(det(C(1:2,1:2,i))));
    M  = 1-arrayfun(f,X,Y);
    imagesc([-4 4],[-4 4],M), colormap gray, axis xy
    axis([-4 4 -4 4])
    axis square
    xlabel('Internal')
    ylabel('Blanket')
    title('Internal and blanket')
    box off
    hold off
    
    subplot(3,3,2)
    f  = @(x,y) exp(-(([x y] - E([3 2],i)')*inv(C([3 2],[3 2],i))*([x; y] - E([3 2],i)))/2)/(2*pi*sqrt(det(C([3 2],[3 2],i))));
    M  = 1-arrayfun(f,X,Y);
    imagesc([-4 4],[-4 4],M), colormap gray, axis xy
    axis([-4 4 -4 4])
    axis square
    xlabel('External')
    ylabel('Blanket')
    title('External and blanket')
    box off
    hold off
    
    subplot(3,3,3)
    f  = @(x,y) exp(-(([x y] - E([1 3],i)')*inv(C([1,3],[1 3],i))*([x; y] - E([1 3],i)))/2)/(2*pi*sqrt(det(C([1 3],[1 3],i))));
    M  = 1-arrayfun(f,X,Y);
    imagesc([-4 4],[-4 4],M), colormap gray, axis xy
    axis([-4 4 -4 4])
    axis square
    xlabel('Internal')
    ylabel('External')
    title('Internal and external')
    box off
    hold off
    drawnow
end

function mmb_memory_plot(f,D,k)
% Plot measures of memory span 
%__________________________________________________________________________
if numel(D) > 1
    for i = 1:numel(D)
        mmb_memory_plot(f,D(i),i/numel(D))
    end
    return
end

if size(D.Ep,2) == 1
    D = mmb_FPE_Laplace(D);
end

if exist('k','var')
    bcol = [0 0 0.9]*(1-k) + [0.7 0.8 0.9]*k;
else
    bcol = [0      0.4470 0.7410];
end

% Figure properties
%--------------------------------------------------------------------------
f.Name = 'Measures of memory';
f.Color = 'w';
f.Position = [500 50 500 550];

% Plot mutual information
%--------------------------------------------------------------------------
MI = zeros(size(D.Cp,3),1);
for i = 1:size(D.Cp,3)
    C = mmb_Schur(D.Cp(:,:,i),2); % Condition upon blanket
    MI(i) = 0.5*log(C(2,2)/det(mmb_Schur(C,1)));
end

subplot(4,1,1)
plot((0:size(D.Ep,2)-1)*D.dt,MI,'Color',bcol), hold on
xlim([0 size(D.Ep,2)*D.dt])
xlabel('Time')
ylabel('nats')
title('Mutual information (conditioned on blanket)')
box off

% Plot precision between internal and external states
%--------------------------------------------------------------------------
P = zeros(size(D.Cp));
for i = 1:size(D.Cp,3)
    P(:,:,i) = inv(D.Cp(:,:,i));
end

subplot(4,1,2)
plot((0:size(P,3)-1)*D.dt,squeeze(P(1,3,:)),'Color',bcol), hold on
xlim([0 size(P,3)*D.dt])
xlabel('Time')
title('Precision')
box off

% Plot information length
%--------------------------------------------------------------------------
subplot(4,1,3)
IL = zeros(size(D.Cp,3),2);
E  = D.Ep;
C  = D.Cp;
for i = 2:size(D.Cp,3)
    IL(i,1) = IL(i-1,1) + sqrt(2*mmb_Laplace_KL(E(:,i),E(:,i-1),C(:,:,i),C(:,:,i-1))); % Information length (full density)
    IL(i,2) = IL(i-1,2) + sqrt(2*mmb_Laplace_KL(E(2,i),E(2,i-1),C(2,2,i),C(2,2,i-1))); % Information length (blanket)
end
plot((0:size(D.Ep,2)-1)*D.dt,IL(:,1)-IL(:,2),'Color',bcol), hold on
xlim([0 size(D.Ep,2)*D.dt])
xlabel('Time')
title('Information length (full density - blanket marginal)')
box off

try
    k;
% Plot reciprocals of real parts of eigenvalues for covariance Jacobian
%--------------------------------------------------------------------------
    u = sort(real(eig(D.Jc)));
    U = -1./u;
    subplot(4,1,4)
    plot(U,'.','MarkerSize',8,'Color',bcol)
    title('Time constants')
    xlabel('Covariance component')
    ylabel('Time')
    box off
    hold on
catch
% Plot reciprocals of real parts of eigenvalues for covariance Jacobian
%--------------------------------------------------------------------------
    u = sort(real(eig(D.Jc)));
    U = -1./u;
    subplot(4,1,4)
    bar(U,'FaceColor',[0.7 0.7 1],'EdgeColor','none')
    title('Time constants')
    xlabel('Covariance component')
    ylabel('Time')
    box off
end

function A = mmb_Schur(B,i)
% Schur complement of element (i,i) in B
%__________________________________________________________________________
C = inv(B);
C(i,:) = [];
C(:,i) = [];
A = inv(C);

function D = mmb_Laplace_KL(E1,E2,C1,C2)
% KL-Divergence between two normal distributions with expectations E and
% covariances C.
%__________________________________________________________________________
D = (trace(C2\C1) + trace((E1-E2)*(E1-E2)'/C2) - length(E1) + log(det(C2)/det(C1)))/2;
D(D<exp(-32)) = 0;

% FUNCTIONS FOR CATEGORICAL DISTRIBUTIONS
%==========================================================================

function D = mmb_Cat_L(d)
% Equip model with transition rate matrix given dissipative and solenoidal
% flows and the steady state distribution.
%__________________________________________________________________________
D   = d;
Q   = D.Q;
R   = D.R;
D.S = D.Sb.*D.Sh.*D.Sm;
D.L = (Q - R)*(diag(D.S(:).^-1));

function D = mmb_Cat_ME(d)
% This function takes a structure, d, comprising the following fields:
% Q        - Solenoidal flow matrix
% R        - Dissipitive flow matrix
% Sh,Sm,Sb - Expectations at NESS
% sh,sm,sb - Initial conditions
% T        - Duration over which to integrate
% dt       - Integration step
% and numerically integrates the associated master equatinon
% equation, returning a structure D whose sh,sm,sb fields includes the 
% value of the probability for every iteration.
%__________________________________________________________________________

% Initialise
%--------------------------------------------------------------------------
D  = d;
T  = D.T;
dt = D.dt;

% Compute probability transition matrix
%--------------------------------------------------------------------------
D   = mmb_Cat_L(D);
L   = D.L;
s   = D.sb.*D.sh.*D.sm;
D.s = s;

% Numerically integrate
%--------------------------------------------------------------------------
for i = 1:round(T/dt)
    s(:)          = s(:) + pinv(L)*(expm(L*dt)-eye(size(L,1)))*L*s(:);
    D.s(:,:,:,i+1)= s;
end

function mmb_Cat_plot(f,D)
% Plotting for the categorical simulation
%__________________________________________________________________________

% Figure properties
%--------------------------------------------------------------------------
f.Name = 'Categorical probabilistic dynamics';
f.Color = 'w';
f.Position = [500 50 500 550];

% Initialise
%--------------------------------------------------------------------------
s  = D.s;
dt = D.dt;
colormap gray

% Plot marginals
%--------------------------------------------------------------------------
subplot(4,1,2)
imagesc((0:size(s,4)-1)*dt,1:3,1-squeeze(sum(s,[2 3]))),caxis([0 1]);
title('External (marginal)')
xlabel('Time')

subplot(4,1,3)
imagesc((0:size(s,4)-1)*dt,1:3,1-squeeze(sum(s,[1 3]))),caxis([0 1]);
title('Internal (marginal)')
xlabel('Time')

subplot(4,1,4)
imagesc((0:size(s,4)-1)*dt,1:3,1-squeeze(sum(s,[1 2]))),caxis([0 1]);
title('Blanket (marginal)')
xlabel('Time')

% Plot evolving distributions
%--------------------------------------------------------------------------
for i = 1:size(s,4)
    subplot(4,3,1)
    imagesc(1-s(:,:,1,i)),caxis([0 1]);
    axis square
    title('Blanket = 1')
    xlabel('Internal')
    ylabel('External')
    
    subplot(4,3,2)
    imagesc(1-s(:,:,2,i)),caxis([0 1]);
    axis square
    title('Blanket = 2')
    xlabel('Internal')
    ylabel('External')
    
    subplot(4,3,3)
    imagesc(1-s(:,:,3,i)),caxis([0 1]);
    axis square
    title('Blanket = 3')
    xlabel('Internal')
    ylabel('External')
    hold off
    drawnow
end

function mmb_Cat_memory_plot(f,D,k)
% Plot measures of memory span (Categorical distributions)
%__________________________________________________________________________
if numel(D) > 1
    for i = 1:numel(D)
        mmb_Cat_memory_plot(f,D(i),i/numel(D))
    end
    return
end

if ~isfield(D,'s')
    D = mmb_Cat_ME(D);
end

if exist('k','var')
    bcol = [0 0 0.9]*(1-k) + [0.7 0.8 0.9]*k;
else
    bcol = [0      0.4470 0.7410];
end

dt = D.dt;

% Figure properties
%--------------------------------------------------------------------------
f.Name = 'Categorical probabilistic dynamics';
f.Color = 'w';
f.Position = [500 50 500 550];

% Plot mutual information
%--------------------------------------------------------------------------
s   = D.s./repmat(sum(D.s,[1 2]+exp(-64)),[3 3 1 1]);                      % Condition upon blanket
shm = repmat(sum(s,2),[1 3 1 1]).*repmat(sum(s,1),[3 1 1 1]);              % Product of marginals
MI  = sum(repmat(sum(D.s,[1 2]),[3 3 1 1]).*s.*(log(s)-log(shm)),[1 2 3]); % Mutual information (averaged over blanket states)
MI  = squeeze(MI);

subplot(3,1,1)
plot((0:size(s,4)-1)*dt,MI,'Color',bcol), hold on
xlim([0 (size(D.s,4)-1)*dt])
xlabel('Time')
ylabel('nats')
title('Mutual information (conditioned on blanket)')
box off

% Plot information length
%--------------------------------------------------------------------------
subplot(3,1,2)
IL = zeros(size(D.s,4),2);

D.s(:,:,:,1) = (D.s(:,:,:,1) + exp(-8))/sum(D.s(:,:,:,1) + exp(-8),[1 2 3]);
for i = 2:size(D.s,4)
    D.s(:,:,:,i) = (D.s(:,:,:,i) + exp(-8))/sum(D.s(:,:,:,i) + exp(-8),[1 2 3]); % Ensure normalisation
    IL(i,1) = IL(i-1,1) + real(sqrt(2*(sum(D.s(:,:,:,i).*(log(D.s(:,:,:,i))-log(D.s(:,:,:,i-1))),[1 2 3]))));                          % Information length (full density)
    IL(i,2) = IL(i-1,2) + real(sqrt(2*sum((sum(D.s(:,:,:,i),[1 2]).*(log(sum(D.s(:,:,:,i),[1 2]))-log(sum(D.s(:,:,:,i-1),[1 2]))))))); % Information length (blanket)
end
plot((0:size(s,4)-1)*dt,IL(:,1)-IL(:,2),'Color',bcol), hold on
xlim([0 (size(D.s,4)-1)*dt])
xlabel('Time')
title('Information length (full density - blanket marginal)')
box off

try
    k;
% Plot reciprocals of real parts of eigenvalues for covariance Jacobian
%--------------------------------------------------------------------------
    u = sort(real(eig(D.L)));
    u(abs(u)<exp(-4)) = [];
    U = -1./u;
    subplot(3,1,3)
    plot(U,'.','MarkerSize',8,'Color',bcol)
    title('Time constants')
    xlabel('Eigenvectors')
    ylabel('Time')
    xlim([1 length(U)])
    box off
    hold on
catch
% Plot reciprocals of real parts of eigenvalues for covariance Jacobian
%--------------------------------------------------------------------------
    u = sort(real(eig(D.L)));
    u(abs(u)<exp(-4)) = [];
    U = -1./u;
    subplot(3,1,3)
    bar(U,'FaceColor',[0.7 0.7 1],'EdgeColor','none')
    title('Time constants')
    xlabel('Eigenvectors')
    ylabel('Time')
    box off
end
