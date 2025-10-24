from MapFeature import *
from FEKFMBL import *

#Import the necessary libraries
import numpy as np
import scipy as sp

class FEKFSLAM(FEKFMBL):
    """
    Class implementing the Feature-based Extended Kalman Filter for Simultaneous Localization and Mapping (FEKFSLAM).
    It inherits from the FEKFMBL class, which implements the Feature Map based Localization (MBL) algorithm using an EKF.
    :class:`FEKFSLAM` extends :class:`FEKFMBL` by adding the capability to map features previously unknown to the robot.
    """
 
    def __init__(self,  *args):

        super().__init__(*args)

        # self.xk_1 # state vector mean at time step k-1 inherited from FEKFMBL

        self.nzm = 0  # number of measurements observed
        self.nzf = 0  # number of features observed

        self.H = None  # Data Association Hypothesis
        self.nf = 0  # number of features in the state vector

        self.plt_MappedFeaturesEllipses = []

        return

    def AddNewFeatures(self, xk, Pk, znp, Rnp):
        """
        This method adds new features to the map. Given:

        * The SLAM state vector mean and covariance:

        .. math::
            {{}^Nx_{k}} & \\approx {\\mathcal{N}({}^N\\hat x_{k},{}^NP_{k})}\\\\
            {{}^N\\hat x_{k}} &=   \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k}}&=
            \\begin{bmatrix}
            {{}^NP_{B}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance

        * And the vector of non-paired feature observations (feature which have not been associated with any feature in the map), and their covariance matrix:

            .. math::
                {z_{np}} &=   \\left[ {}^Bz_{F_1} ~  \\cdots ~ {}^Bz_{F_{n_{zf}}}  \\right]^T \\\\
                {R_{np}}&= \\begin{bmatrix}
                {}^BR_{F_1} &  \\cdots & 0  \\\\
                \\vdots &  \\ddots & \\vdots \\\\
                0 & \\cdots & {}^BR_{F_{n_{zf}}}
                \\end{bmatrix}
                :label: FEKFSLAM-non-paire-feature-observations

        this method creates a grown state vector ([xk_plus, Pk_plus]) by adding the new features to the state vector.
        Therefore, for each new feature :math:`{}^Bz_{F_i}`, included in the vector :math:`z_{np}`, and its corresponding feature observation noise :math:`{}^B R_{F_i}`, the state vector mean and covariance are updated as follows:

            .. math::
                {{}^Nx_{k}^+} & \\approx {\\mathcal{N}({}^N\\hat x_{k}^+,{}^NP_{k}^+)}\\\\
                {{}^N x_{k}^+} &=
                \\left[ {{}^N x_{B_k}^T} ~ {{}^N x_{F_1}^T} ~ \\cdots ~{{}^N x_{F_n}^T}~ |~\\left({{}^N x_{B_k} \\boxplus ({}^Bz_{F_i} }+v_k)\\right)^T \\right]^T \\\\
                {{}^N\\hat x_{k}^+} &=
                \\left[ {{}^N\\hat x_{B_k}^T} ~ {{}^N\\hat x_{F_1}^T} ~ \\cdots ~{{}^N\\hat x_{F_n}^T}~ |~{{}^N\\hat x_{B_k} \\boxplus {}^Bz_{F_i}^T } \\right]^T \\\\
                {P_{k}^+}&= \\begin{bmatrix}
                {{}^NP_{B_k}}  &  {{}^NP_{B_kF_1}}   &  \\cdots   &  {{}^NP_{B_kF_n}} & | & {{}^NP_{B_k} J_{1 \\boxplus}^T}\\\\
                {{}^NP_{F_1B_k}}  &  {{}^NP_{F_1}}   &  \\cdots   &  {{}^NP_{F_1F_n}} & | & {{}^NP_{F_1B_k} J_{1 \\boxplus}^T}\\\\
                \\vdots  & \\vdots & \\ddots  & \\vdots & | &  \\vdots \\\\
                {{}^NP_{F_nB_k}}  &  {{}^NP_{F_nF_1}}   &  \\cdots   &  {{}^NP_{F_n}}  & | & {{}^NP_{F_nB_k} J_{1 \\boxplus}^T}\\\\
                \\hline
                {J_{1 \\boxplus} {}^NP_{B_k}}  &  {J_{1 \\boxplus} {}^NP_{B_kF_1}}   &  \\cdots   &  {J_{1 \\boxplus} {}^NP_{B_kF_n}}  & | &  {J_{1 \\boxplus} {}^NP_R J_{1 \\boxplus} ^T} + {J_{2\\boxplus}} {{}^BR_{F_i}} {J_{2\\boxplus}^T}\\\\
                \\end{bmatrix}
                :label: FEKFSLAM-add-a-new-feature

        :param xk: state vector mean
        :param Pk: state vector covariance
        :param znp: vector of non-paired feature observations (they have not been associated with any feature in the map)
        :param Rnp: Matrix of non-paired feature observation covariances
        :return: [xk_plus, Pk_plus] state vector mean and covariance after adding the new features
        """
        #assert znp.size > 0, "AddNewFeatures: znp is empty"
        ## To be completed by the student

        xk_1_R = Pose3D(xk[0:self.xB_dim])   # Extract robot's pose
        Pk_1_robot = Pk[0:self.xB_dim, 0:self.xB_dim]       # Extract covariance of the robot
 
        Pk_plus = Pk.copy()  # Copy the original covariance matrix (to be expanded)
        xk_plus = xk.copy()  # Copy the original state vector (to be expanded)
   
        #Calculate number of features (nf) already in the state vector.
        #The state vector contains robot state (xB_dim) and features (xF_dim each).
        nf = int((Pk.shape[0] - self.xB_dim) / self.xF_dim)

        # loop over the new non-paired features
        for i in range(len(znp)):

            #Adding a New Feature to the State Vector
            NxFi = self.g(xk_1_R, znp[i])  
            xk_plus = np.block([[xk_plus], [NxFi]])

           
            #Updating the Covariance Matrix
            # (1) Compute Feature Covariance of i(red)
            Jgxi = self.Jgx(xk_1_R, znp[i])  # Jacobian w.r.t. state
            Jgvi = self.Jgv(xk_1_R, znp[i])  # Jacobian w.r.t. noise
            NpFi = Jgxi @ Pk_1_robot @ Jgxi.T + Jgvi @ Rnp[i] @ Jgvi.T

            #(2) Split the Covariance Matrix for Expansion yellow
            A = Pk_plus  # Original covariance matrix
            B1 = np.split(Pk[0:self.xB_dim, :], [self.xB_dim], axis=1)  # Split covariance
            
            # (3) Compute Cross-Covariance (B) green
            B = np.block([
                Jgxi @ B1[0],  # Cross-covariance between new feature and robot
                Jgxi @ B1[1]   # Cross-covariance between new feature and existing features
            ])

            # (4) Update the Covariance Matrix
            C = NpFi  # New feature’s covariance matrix
            Pk_plus = np.block([[A, B.T], [B, C]])  # Expand covariance matrix

            # Updates Pk and increments the number of mapped features (nf).
            Pk = Pk_plus  # Update `Pk` with new covariance matrix
            self.nf += 1  # Increase feature count

        return xk_plus, Pk_plus

    
    def Prediction(self, uk, Qk, xk_1, Pk_1):
        """
        This method implements the prediction step of the FEKFSLAM algorithm. It predicts the state vector mean and
        covariance at the next time step. Given state vector mean and covariance at time step k-1:

        .. math::
            {}^Nx_{k-1} & \\approx {\\mathcal{N}({}^N\\hat x_{k-1},{}^NP_{k-1})}\\\\
            {{}^N\\hat x_{k-1}} &=   \\left[ {{}^N\\hat x_{B_{k-1}}^T} ~ {{}^N\\hat x_{F_1}^T} ~  \\cdots ~ {{}^N\\hat x_{F_{nf}}^T} \\right]^T \\\\
            {{}^NP_{k-1}}&=
            \\begin{bmatrix}
            {{}^NP_{B_{k-1}}} & {{}^NP_{BF_1}} & \\cdots & {{}^NP_{BF_{nf}}}  \\\\
            {{}^NP_{F_1B}} & {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_{nf}}}  \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_{nf}B}} & {{}^NP_{F_{nf}F_1}} & \\cdots & {{}^NP_{nf}}  \\\\
            \\end{bmatrix}
            :label: FEKFSLAM-state-vector-mean-and-covariance-k-1

        the control input and its covariance :math:`u_k` and :math:`Q_k`, the method computes the state vector mean and covariance at time step k:

        .. math::
            {{}^N\\hat{\\bar x}_{k}} &=   \\left[ {f} \\left( {{}^N\\hat{x}_{B_{k-1}}}, {u_{k}}  \\right)  ~  { {}^N\\hat x_{F_1}^T} \\cdots { {}^N\\hat x_{F_n}^T}\\right]^T\\\\
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}
            :label: FEKFSLAM-prediction-step

        where

        .. math::
            {F_{1_k}} &= \\left.\\frac{\\partial {f_S({}^Nx_{k-1},u_k,w_k)}}{\\partial {{}^Nx_{k-1}}}\\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}} \\\\
             &=
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{B_{k-1}}}} &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {{}^Nx_{Fn}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}} \\\\
            \\vdots & \\vdots & \\ddots & \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{k-1}}} &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{F1}}} &
            \\cdots &
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {{}^Nx_{Fn}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_x}} & {0} & \\cdots & {0} \\\\
            {0}   & {I} & \\cdots & {0} \\\\
            \\vdots& \\vdots  & \\ddots & \\vdots  \\\\
            {0}   & {0} & \\cdots & {I} \\\\
            \\end{bmatrix}
            \\\\{F_{2_k}} &= \\left. \\frac{\\partial {f({}^Nx_{k-1},u_k,w_k)}}{\\partial {w_{k}}} \\right|_{\\begin{subarray}{l} {{}^Nx_{k-1}}={{}^N\\hat x_{k-1}} \\\\ {w_k}={0}\\end{subarray}}
            =
            \\begin{bmatrix}
            \\frac{\\partial {f} \\left( {{}^Nx_{B_{k-1}}}, {u_{k}}, {w_{k}}  \\right)}{\\partial {w_{k}}} \\\\
            \\frac{\\partial {{}^Nx_{F1}}}{\\partial {w_{k}}}\\\\
            \\vdots \\\\
            \\frac{\\partial {{}^Nx_{Fn}}}{\\partial {w_{k}}}
            \\end{bmatrix}
            =
            \\begin{bmatrix}
            {J_{f_w}}\\\\
            {0}\\\\
            \\vdots\\\\
            {0}\\\\
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-Jacobian

        obtaining the following covariance matrix:
        
        .. math::
            {{}^N\\bar P_{k}}&= {F_{1_k}} {{}^NP_{k-1}} {F_{1_k}^T} + {F_{2_k}} {Q_{k}} {F_{2_k}^T}{{}^N\\bar P_{k}}
             &=
            \\begin{bmatrix}
            {J_{f_x}P_{B_{k-1}} J_{f_x}^T} + {J_{f_w}Q J_{f_w}^T}  & |  &  {J_{f_x}P_{B_kF_1}} & \\cdots & {J_{f_x}P_{B_kF_n}}\\\\
            \\hline
            {{}^NP_{F_1B_k} J_{f_x}^T} & |  &  {{}^NP_{F_1}} & \\cdots & {{}^NP_{F_1F_n}}\\\\
            \\vdots & | & \\vdots & \\ddots & \\vdots \\\\
            {{}^NP_{F_nB_k} J_{f_x}^T} & | &  {{}^NP_{F_nF_1}} & \\cdots & {{}^NP_{F_n}}
            \\end{bmatrix}
            :label: FEKFSLAM-prediction-step-covariance

        The method returns the predicted state vector mean (:math:`{}^N\\hat{\\bar x}_k`) and covariance (:math:`{{}^N\\bar P_{k}}`).

        :param uk: Control input
        :param Qk: Covariance of the Motion Model noise
        :param xk_1: State vector mean at time step k-1
        :param Pk_1: Covariance of the state vector at time step k-1
        :return: [xk_bar, Pk_bar] predicted state vector mean and covariance at time step k
        """

       ## To be completed by the student
        #Handling Optional Inputs: If None values, keeps the last stored values from the previous prediction step.
        self.xk_1 = xk_1 if xk_1 is not None else self.xk_1
        self.Pk_1 = Pk_1 if Pk_1 is not None else self.Pk_1

        # Logging Control Input & Motion Noise
        self.uk = uk
        self.Qk = Qk 

        #Computing Predicted State Vector 
        xk_bar = xk_1.copy()  
        xk_bar[0:3] = self.f(xk_1[0:3], uk)  

        # Initializing the New Covariance Matrix
        pk_bar = Pk_1.copy()
        xk_1_R = Pose3D(xk_1[0:3]) 

        # Computing Jacobians
        Ak = self.Jfx(xk_1_R)  # Jacobian of motion model w.r.t. state
        Wk = self.Jfw(xk_1_R)  # Jacobian of motion model w.r.t. noise

        # Computing Covariance Updates
        N_Pk = Pk_1[0:3, 0:3]  # Extract robot's covariance
        A = Ak @ N_Pk @ Ak.T + Wk @ Qk @ Wk.T  # Compute new covariance

        # Computing Cross-Covariance
        B = Ak @ Pk_1[0:3, 3:]  # Compute cross-covariance between robot and features
        #Updates pk_bar with cross-covariance values.
        pk_bar[0:3, 3:] = B  
        pk_bar[3:, 0:3] = B.T 

        # Completing Covariance Update
        C = Pk_1[3:, 3:]  # Extract previous feature covariance
        Pk_1 = np.block([[A, B], [B.T, C]])  # Assemble new covariance matrix

        #Storing Results & Returning Predictions
        self.xk_bar = xk_bar
        self.Pk_bar = Pk_1
        return self.xk_bar, self.Pk_bar
        
    def Localize(self, xk_1, Pk_1):
        """
        This method implements the FEKFSLAM algorithm. It localizes the robot and maps the features in the environment.
        It implements a single interation of the SLAM algorithm, given the current state vector mean and covariance.
        The unique difference it has with respect to its ancestor :meth:`FEKFMBL.Localize` is that it calls the method
        :meth:`AddNewFeatures` to add new non-paired features to the map.

        :param xk_1: state vector mean at time step k-1
        :param Pk_1: covariance of the state vector at time step k-1
        :return: [xk, Pk] state vector mean and covariance at time step k
        """
        ## To be completed by the student
        #---------------------------------------------------------------
        # prediction - part 1
        # uk, Qk = self.GetInput()
        # xk_bar, Pk_bar = self.Prediction(uk, Qk, xk_1, Pk_1)
        # self.xk, self.Pk = xk_bar, Pk_bar
        #---------------------------------------------------------------

        # Prediction Step (Motion Update) Part 2     
        uk, Qk = self.GetInput()
        xk_bar, Pk_bar = self.Prediction(uk, Qk, xk_1, Pk_1)

        # Measurement Update (Feature Observation Update
        zm, Rm, Hm, Vm = self.GetMeasurements()
        zf, Rf = self.GetFeatures()

        self.H = self.DataAssociation(xk_bar, Pk_bar, zf, Rf)

        zk, Rk, Hk, Vk, znp, Rnp = self.StackMeasurementsAndFeatures(xk_bar, zm, Rm, Hm, Vm, zf, Rf, self.H)
        
        xk, Pk = self.Update(zk, Rk, xk_bar, Pk_bar, Hk, Vk)
        
        # Feature Mapping (Adding New Features)
        # xk, Pk = self.AddNewFeatures(xk_bar,Pk_bar,znp,Rnp)     # Part 1 without updates
        if len(znp) > 0:
            xk, Pk = self.AddNewFeatures(xk,Pk,znp,Rnp)         # Part 2 with updates

        # Storing the Updated State Vector and Covariance Matrix    
        self.xk, self.Pk = xk, Pk 

        #---------------------------------------------------------------
        # Use the variable names zm, zf, Rf, znp, Rnp so that the plotting functions work
        self.Log(self.robot.xsk, self.GetRobotPose(self.xk), self.GetRobotPoseCovariance(self.Pk),
                 self.GetRobotPose(self.xk_bar), zm)  # zm - > NONE for prediction - log the results for plotting

        # convert all the arguments passed to PlotUncertainty (from list to arr)
        # except znp and Rnp = which are alr an arr in SplitFeatures 
        Fj= len(zf) * self.xF_dim
        zf = np.array(zf).reshape(Fj,1)        #12x1
        
        # Rf  - > cov 0 for every ft obs = 12x12

        R_plot = np.zeros([0,0])
        for i in range(zf.shape[0] - len(Rf)):
            R_plot=scipy.linalg.block_diag(R_plot,Rf[i])
        Rf= np.array(R_plot)

        # -----------------------------------------
        # Fnj= len(znp) * self.xF_dim
        # znp = np.array(znp).reshape(Fnj,1)  # _x1

        # Rnp_plot = np.zeros([0,0])     #_x_
        # for i in range(zf.shape[0] - len(Rf)):
        #     Rnp_plot=scipy.linalg.block_diag(Rnp_plot,Rf[i])
        # Rnp= np.array(Rnp_plot)
        # -----------------------------------------

        self.PlotUncertainty(zf, Rf, znp, Rnp)
        # self.PlotRobotUncertainty()       #for only prediction

        return self.xk, self.Pk

    def LocalizationLoop(self, x0, P0, usk):
        """
        Localization loop. During *self.kSteps* it calls the :meth:`Localize` method for each time step.

        :param x0: initial state vector
        :param P0: initial covariance matrix
        """

        xk_1 = x0
        Pk_1 = P0

        xsk_1 = self.robot.xsk_1

        zf,Rf = self.GetFeatures()
        
      
        xk_1, Pk_1 = self.AddNewFeatures(xk_1,Pk_1,zf,Rf)

        for self.k in range(self.kSteps):
            xsk = self.robot.fs(xsk_1, usk)  # Simulate the robot motion
            xk, Pk = self.Localize(xk_1, Pk_1)  # Localize the robot

            xsk_1 = xsk  # current state becomes previous state for next iteration
            xk_1 = xk
            Pk_1 = Pk
            
        self.PlotState()  # plot the state estimation results
        plt.show()

    def PlotMappedFeaturesUncertainty(self):
        """
        This method plots the uncertainty of the mapped features. It plots the uncertainty ellipses of the mapped
        features in the environment. It is called at each Localization iteration.
        """
        # remove previous ellipses
        for i in range(len(self.plt_MappedFeaturesEllipses)):
            self.plt_MappedFeaturesEllipses[i].remove()
        self.plt_MappedFeaturesEllipses = []

        self.xk=BlockArray(self.xk,self.xF_dim, self.xB_dim)
        self.Pk=BlockArray(self.Pk,self.xF_dim, self.xB_dim)

        # draw new ellipses
        for Fj in range(self.nf):
            feature_ellipse = GetEllipse(self.xk[[Fj]],
                                         self.Pk[[Fj,Fj]])  # get the ellipse of the feature (x_Fj,P_Fj)
            plt_ellipse, = plt.plot(feature_ellipse[0], feature_ellipse[1], 'r')  # plot it
            self.plt_MappedFeaturesEllipses.append(plt_ellipse)  # and add it to the list

    def PlotUncertainty(self, zf, Rf, znp, Rnp):
        """
        This method plots the uncertainty of the robot (blue), the mapped features (red), the expected feature observations (black) and the feature observations.

        :param zf: vector of feature observations
        :param Rf: covariance matrix of feature observations
        :param znp: vector of non-paired feature observations
        :param Rnp: covariance matrix of non-paired feature observations
        :return:
        """
        if self.k % self.robot.visualizationInterval == 0:
            self.PlotRobotUncertainty()
            self.PlotFeatureObservationUncertainty(znp, Rnp,'b')
            self.PlotFeatureObservationUncertainty(zf, Rf,'g')
            self.PlotExpectedFeaturesObservationsUncertainty()
            self.PlotMappedFeaturesUncertainty()