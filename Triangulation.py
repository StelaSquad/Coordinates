import numpy as np


class Triangulate():
    """When coordinate systems are spun about an axis, they are first spun about the z-axis
    by angle alpha and then"""
    def __init__(self):
        self.comp_power = 100
        self.chain=[]
        self.dist=[]
        self.dth=0
        self.dph=0
        self.obserrs = [np.pi/180/60,np.pi/180/60]

    
    def xp(self,a,b):
        """Returns x-prime, the vector representing the primed x-axis in the celestial frame. 
        Points toward the sky, normal to surface of earth.
        
        Input:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
        Output:
            ------
            x_prime: ndarray [arg, dtype=float, shape=(3)
                3-vector that represents the x-axis of the prime frame in celestial xyz coordinates.
        """
        
        x_prime = np.array([np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),np.sin(b)])
        return x_prime

    def yp(self,a,b):
        """Returns y-prime, the vector representing the primed x-axis in the celestial frame. 
        Points toward the east, tangent to surface of earth.
        
        Input:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
        Output:
            ------
            y_prime: ndarray [arg, dtype=float, shape=(3)
                3-vector that represents the y-axis of the prime frame in celestial xyz coordinates.
        """
    
        y_prime = np.array([-np.sin(a),np.cos(a),0])
        return y_prime
    
    def zp(self,a,b):
        """Returns z-prime, the vector representation of the primed z-axis in the celestial frame.
        Points north, tangent to surface of earth
        
        Input:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
        Output:
            ------
            z_prime: ndarray [arg, dtype=float, shape=(3)
                3-vector that represents the z-axis of the prime frame in celestial xyz coordinates.
        """

        z_prime = np.array([-np.cos(a)*np.sin(b),-np.sin(a)*np.sin(b),np.cos(b)])
        return z_prime


    def vec_prime(self, a, b, v, form='xyz'):
        """Transforms a celestial-coordinate vector into a primed frame representation.
        
        Inputs:
            ------
            a: float 
                Alpha, the angle that the primed coordinate system is spun about its z-axis.
               
            b: float
                Beta, angle that the primed coordinate system is pitched about its y-axis.
                
            v: ndarray or list [arg, dtype = float, shape=(2) or shape=(3)]
                A celestial vector, in either a xyz or theta-phi representation.
                
        Optional:
            ------
            rep: string, arg = 'xyz' or 'th-ph'
                Specify whether the output data should be in xyz coordinates or theta-phi coordinates. 
                
        
        Output:
            v_prime: ndarray [arg, dtype=float, shape=(3) or shape=(2)]
                The vector represented in the primed coordinate system.
        
        """
        
        v = np.array(v)
        
        if len(v) == 2:
            v = self.xyz(v[0],v[1])
            
            
        x = np.dot(self.xp(a,b),v)
        z = np.dot(self.zp(a,b),v)
        y = np.dot(self.yp(a,b),v)
        
        v_xyz = np.round(np.array([x,y,z]),12)

        
        if form == 'xyz':
            v_out = v_xyz
            
        elif form == 'th-ph':
            th = np.arctan2(v_xyz[1],v_xyz[2])
            ph = np.arcsin(v_xyz[0])
            v_out = np.array([th,ph])
            
        else:
            raise ValueError("Requested representation not understood. Use either 'xyz' or 'th-ph")
            
        return v_out


    def xyz(self,th,ph):
        """Transforms spherical coordinates into xyz coordinates in the celestial frame."""
        return np.array([np.cos(th)*np.cos(ph), np.sin(th)*np.cos(ph), np.sin(ph)])
    
    
    def gen_mock(self,errs=[0,0]):
        
        """Generates a mock triangulation data set, to test the program.
                
        Optional:
            ------
            errs: list or ndarray [args, dtype=float shape=(2)] 
                list the error in the the generated data of the observed angle differences
                
                
        
        Output:
            v_prime: ndarray [arg, dtype=float, shape=(3) or shape=(2)]
                The vector represented in the primed coordinate system.
        """
        
        # Generate random truth values
        a = np.random.rand()*2*np.pi
        b = (np.random.rand()-0.5)*np.pi
        
        # Generate three random triangulation vectors
        v1 = [np.random.uniform(0,2*np.pi),np.random.uniform(-np.pi/2,np.pi/2)]
        v2 = [np.random.uniform(0,2*np.pi),np.random.uniform(-np.pi/2,np.pi/2)]
        v3 = [np.random.uniform(0,2*np.pi),np.random.uniform(-np.pi/2,np.pi/2)]
        
        # Calculate the true difference in alt-azimuth, and add a little random error
        v2_v1 = self.vec_prime(a,b,v2,form='th-ph') - self.vec_prime(a,b,v1,form='th-ph') + np.random.uniform(-errs[0],errs[0])
        v3_v2 = self.vec_prime(a,b,v3,form='th-ph') - self.vec_prime(a,b,v2,form='th-ph') + np.random.uniform(-errs[1],errs[1])
                
        
        return [a,b],v1,v2,v3,v2_v1,v3_v2

    
    

    def find_valid(self, obj1coor, obj2coor, obs, lims=[[0,2*np.pi],[-np.pi/2,np.pi/2]]):
        """Calculates the probability distribution of lattitude and longitude points within the given limits.
        
        
        Inputs:
            ------
            obj1coor: list or ndarray [args, dtype=float, shape=(2)]
                Celestial angle coordinates for the first object
                
            obj2coor: list or ndarray [args, dtype=float, shape=(2)]
                Celestial angle coordinates for the second object
                
            obs: list or ndarray [args, dtype=float, shape=(2)]
                the observed difference in [azimuth, altitude] between object 1 and 2
                
        Optional:
            ------
            lims: list or ndarray [args, dtype=float, shape=(2,2)]
                The limits in which to look for valid longitude and lattitudes. Default is the entire space.
            
        
        Output:
            ------
            grid: list [args,dtype=float,shape=(2,100,100)]
                The coordinates of each longitude and lattitude tested
            
            dist_norm: ndarray [args, dtype=float, shape=(100,100)]
                The normalized probability distribution over all points in the grid.
                
        """
            
        n=self.comp_power
    
        obsth = obs[0]
        obsph = obs[1]
        
        # Initalize variable space
        A = np.linspace(lims[0][0],lims[0][1],n)
        B = np.linspace(lims[1][0],lims[1][1],n)
        grid = np.meshgrid(A,B)
        
        flat = np.array([grid[0].flatten(),grid[1].flatten()]).T
            
        # calculate observation vectors
        obj1_xyz = self.xyz(obj1coor[0],obj1coor[1]) # xyz(obj1[0],obj1[1])
        obj2_xyz = self.xyz(obj2coor[0],obj2coor[1]) # (obj2[0],obj2[1])


        # For each potential A and B vector, calculate the theoretical change in theta and phi 
        thp = []
        php = []
        for x in flat:
            # Calculate the vectors in a frame [alpha, beta] on the surface of the earth
            [thp1,php1] = self.vec_prime(x[0],x[1],obj1_xyz,form='th-ph')
            [thp2,php2] = self.vec_prime(x[0],x[1],obj2_xyz,form='th-ph')

            # Calculate theoretical difference between the two angles
            thp += [thp2-thp1]
            php += [php2-php1]

        #back from column to 2d grid
        thp= np.array(thp).reshape(n,n)
        php= np.array(php).reshape(n,n)

        # Create surface that represents the True change in alt-az coords (as observed)
        obs12_th = np.ones((n,n))*obsth
        obs12_ph = np.ones((n,n))*obsph

        # Take the observed delta-theta and delta-phi and compare it to our theoretical ones to figure out
        # which values of A and B would allow for the observed changes.
        
        # Set up empty array for output valeues
        sel_fin = np.array([])
        
        # Start real small with the binsize, extremely restrictive
        stdth = np.std(thp.flatten())
        stdph = np.std(php.flatten())*2
        
        #width = (max(thp.flatten())-min(thp.flatten()))/2
        
        mod=0.01
        dist = np.ones((n,n))*10**-12
        
        
        while mod*stdth < 3*self.obserrs[0] and mod*stdph < 3*self.obserrs[1]:
            
            mod*=1.01
            
        thdist = np.exp(- ((thp-obs12_th)/(mod*stdth))**2 )
        phdist = np.exp(- ((php-obs12_ph)/(mod*stdph))**2 )
                
        #thdist = np.exp(- ((thp-obs12_th)/(self.obserrs[0]))**2 )
        #phdist = np.exp(- ((php-obs12_ph)/(self.obserrs[1]))**2 )        
                
        dist = (thdist/np.sum(thdist))*(phdist/np.sum(phdist))
        dist_norm = dist/np.sum(dist)
        
        self.chain+=[dist]
        
        return grid,dist_norm
    
    def match(self,grid, c1,c2,c3):
        """Combines three probability distributions to isolate points for which the lattitude and longitude values match observation"""
        
        # We want to compare two functions whose data points are not necessarily the same. Thus, we have to bin both into
        # a new, global data set.
        
        comb = c1*c2*c3
        comb /= np.sum(comb)
                
        self.dist+=[[grid,comb]]
        
        thn = np.sum(comb,axis=0)
        phn = np.sum(comb,axis=1)
        
        th_ax = grid[0][0]
        ph_ax = grid[1][:,0]
        
        th_avg = np.sum(thn*th_ax)
        ph_avg = np.sum(phn*ph_ax)
        
        #print sum(thn),sum(phn)
        Np = np.sum(thn > 0.001)
        n = len(thn)
        
        th_std = np.sqrt(np.sum( (th_ax - th_avg)**2 * thn))# * Np/(Np-1) )/ np.sqrt(n)
        ph_std = np.sqrt(np.sum( (ph_ax - ph_avg)**2 * phn))# * Np/(Np-1) )/ np.sqrt(n)
        
        
        return np.array([[th_avg,ph_avg],[th_std,ph_std]])

    
    def triangulate(self, v1, v2, v3, obs_v1_v2, obs_v2_v3, iterations = 5, obserrs=[None,None]):
        """given v1,v2,v3, three celestial coordinates for objects, and observed changes in altitude and azimuth of those
        objects, returns exact coordinates of normal vector. i.e. latitude and longitude"""
        
        if sum(np.array(obserrs)==None) == 0:
            self.obserrs = obserrs
        
        
        # Calculate difference between v1 and v3
        obs_v1_v3 = [obs_v1_v2[0]+ obs_v2_v3[0], obs_v1_v2[1]+ obs_v2_v3[1]]
        
        lims = [[0,2*np.pi],[-np.pi/2,np.pi/2]]
        
        
        for i in range(iterations):
            print "Running for lims: " + str(np.round(lims,5).tolist())
            
            # find the probability distributions for each observation
            grid, c1 = self.find_valid(v1, v2, obs_v1_v2, lims=lims)
            _, c2 = self.find_valid(v1, v3, obs_v1_v3, lims=lims)
            _, c3 = self.find_valid(v2, v3, obs_v2_v3, lims=lims)
            
            
            if np.sum(np.isnan(c1*c2*c3) ==0):
                
                # Matches all three
                [av,acc] = self.match(grid,c1,c2,c3)
                
                
                # Finds the accuracy of the analysis, chooses new limits based on these
                r = 5
                dth = grid[0][0][1]-grid[0][0][0]
                dph = grid[1][1][0]-grid[1][0][0]
                
                acc += np.array([dth,dph])/(r)
                
                lims = np.array([av - r*acc, av + r*acc]).T
                
                
            else:
                print "minimum value reached"
                break
                        
            
        print "Done."
        return av,acc