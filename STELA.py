import numpy as np
import astroquery.simbad as aq
import astropy.coordinates as cp
import astropy.time as time
import astropy.units as u
from astropy.table import Table
import os

class STELA():
    """
    Class that governs the alignment of telescope, star positions, catalogs, and star identification
    
    Quantities and objects:
        STELA.naked: catolog of nearest stars, brightes in sky
    
    Functions:
        STELA.setup_cats:
            Setup catolog files.
        STELA.get_ref_stars:
            Given an estimated latitude and longitude, returns the three brightest stars in the sky to align.
        STELA.gen_mock_obs:
            For testing triangulation
        STELA.triangulate:
            After calling get_ref_stars, and measuring the differences in alt-az coordinates of the three
            points, STELA.triangulate can locate the new latitude and longitude that accounts for the error
            in telescope positioning.
        
    """
    
    def __init__(self):
        self.catolog = aq.Simbad()
        self.catolog.remove_votable_fields('coordinates')
        self.catolog.add_votable_fields('ra','dec','flux(V)')
        
        self.reset_cats = False
        
        self.triangulation_class = Triangulate()
        
        self.setup_cats()
    
    def setup_cats(self):
        """ 
        Sets up the necessary catalogs, prints them to a file. (No parameters)
        
        """
        
        if self.reset_cats == True:
            os.system('rm -r catalog')
            
        if os.path.exists('./catalog/naked.dat') == False:
            if os.path.exists('catalog') == False:
                os.mkdir('catalog')

            self.naked =  self.catolog.query_catalog('GJ')
            
            select = np.ones(len(self.naked),dtype='bool')
            select[np.where(np.isnan(np.array(self.naked['FLUX_V'])))[0]] = False
                
            self.naked = self.naked[select]
            self.naked = Table(np.unique(self.naked))
            self.naked.sort("FLUX_V")
            
            self.naked.write('./catalog/naked.dat',format='ascii')
        else:
            
            self.naked = Table.read('./catalog/naked.dat',format='ascii')
            

    
    def get_ref_stars(self,lon_est,lat_est,unit=[u.deg,u.deg]):
        """
        Given an estimation of longitude and latitude, identifies 3 target stars to use as 
        triangulation coordinates
        
        Input:
            ------
            lon_est: float
                The current longitude at which the telescope is set up
            
            lat_est: float
                The current latitude at which the telescope is set up
            
        Output:
            ------
            altaz_calib: list [args, shape=(3,2), dtype=float]
                The estimated positions in the altitude azimuth coordinate frame. Can be used to point
                telescope to approximate position of stars.
        """
        
        
        self.lat_est = lat_est*unit[0]
        self.lon_est = lon_est*unit[1]
        
        # Set observation time for calibration
        self.time = time.Time.now()

        # Set up approximate earth frame
        loc_est = cp.EarthLocation(lon_est*unit[0],lat_est*unit[1])
        earth_n = cp.AltAz(0*u.deg,90*u.deg,location=loc_est,obstime=self.time)
        
        # Coordinates of all visible stars
        ra= self.naked["RA"].data
        dec = self.naked["RA"].data
        cat = cp.SkyCoord(ra = ra,dec = dec,unit=[u.hourangle,u.deg])

        # Choose stars that are at least 30 degrees above horizon
        select = earth_n.separation(cat) < cp.Angle(60*u.deg)

        # Select the first three
        c=0
        inds=[]
        for i in range(len(select)):
            if select[i] == True:
                c+=1
                inds+=[i]
                if c>=3:
                    break

        # choose the three ones from the catalog
        cel_calib= self.naked.copy()
        cel_calib.remove_rows(range(len(cel_calib)))
        
        for i in inds:
            cel_calib.add_row(self.naked[i])
            
        # celestial coordinate points
        cel_coors = [cel_calib["RA"],cel_calib["DEC"]]
        
        # coordinates in earth frame
        p_coors = cp.SkyCoord(cel_coors[0],cel_coors[1],unit=[u.hourangle,u.deg])
        altaz_calib = p_coors.transform_to(earth_n)
        
        # set class objects
        self.altaz_calib = altaz_calib
        self.cel_calib = cel_calib
        
        return altaz_calib

    
    def gen_mock_obs(self):
        """
        Generates fake observation data based on possible errors of +/- 5deg.
        
        Output:
            ------
            obs: list [args,len=2,dtype=float]
                Difference in altitude and azimuth for three points, used in triangulation.
        
        """
        
        # creates class object based on random errors in telescope placement
        self.mock_home = [self.lon_est + np.random.uniform(-5,5)*u.deg,
                          self.lat_est + np.random.uniform(-5,5)*u.deg]
        
        # creates objects based on errors
        mock_loc = cp.EarthLocation(self.mock_home[0],self.mock_home[1])
        surf = cp.AltAz(location=mock_loc,obstime=self.time)
        pts = self.altaz_calib.transform_to(surf)
        
        # calculate angular differences in altitude and azimuth for points
        self.v2_v1 = [pts[1].az.rad - pts[0].az.rad, pts[1].alt.rad - pts[0].alt.rad]
        self.v3_v2 = [pts[2].az.rad - pts[1].az.rad, pts[2].alt.rad - pts[1].alt.rad]
        
        return [self.v2_v1, self.v3_v2]
    
    def triangulate(self, v2_v1,v3_v2):
        """
        Used to triangulate the true latitude and longitude corresponding to the norm of the telescope position.
        
        Input
            ------
            v2_v1: list [args,len=2,dtype=float]
                The difference in [azimuth,altitude] between object 2 and object 1
               
            v3_v2: list [args,len=2,dtype=float]
                The difference in [azimuth,altitude] between object 3 and object 2
        """
        
        ra = cp.Angle(self.cel_calib["RA"], unit=u.hourangle)
        dec = cp.Angle(self.cel_calib["DEC"], unit=u.deg)

        v = np.array([ra.rad,dec.rad]).T

        out = self.triangulation_class.triangulate(v[0],v[1],v[2],v2_v1,v3_v2)

        n = cp.SkyCoord(out[0][0],out[0][1],unit=u.rad,frame='icrs')

        npr = cp.EarthLocation(lon=0*u.deg,lat=90*u.deg,height=0*u.m)

        n.location=npr
        n.obstime=self.time

        [self.lon,self.lat] = [180-n.altaz.az.deg, n.altaz.alt.deg]
        self.home = [self.lon,self.lat]
        
        h = cp.EarthLocation(self.home[0]*u.deg,self.home[1]*u.deg)
        self.tel_frame = cp.AltAz(location=h,obstime=self.time)
        v3 = cp.SkyCoord(v[2][0],v[2][1],unit=u.rad)
        self.tel_pos = v3

        
        #return [self.on,self.lat]
    
    
    
    
    
    

class Triangulate():
    """
    This class is used to triangulate the location of the telescope based on three celestial objects. 
    
    It operates based on a latitude and longitude equatorial coordinate system, with celestial coordinates
    given in that frame. The telescope can be placed a floor that is not perfectly normal to the surface 
    of the earth. The use a (technically) false latitude and longitude, one for which the normal vector is the
    same as the normal vector of the base of the telescope, can account for this error. This is the method
    used, but note that the outputted earth coordinates will likely differ slightly than the true location of
    the telescope.
    
    Objects:
        Triangulate.obserrs: 
                Expected errors in measured altitude and azimuth (default=1')
        Triangulate.comp_power: T
                he calculative potential of the computer being used (default=100)
    
    Functions:
        Triangulate.triangulate:
                Computes the normal vector of the telescope based on three objects and their observed angular
                differences.
        
    Helper Functions:
        Triangulate.xp, Triangulate.yp, Triangulate.zp: 
                The coordinate vectors of a frame on the surface of the earth, at location [a,b].
        Triangulate.vec_prime: 
                Transforms a vector in equatorial frame into an earth alt-az frame at location [a,b].
        Triangulate.find_valid: 
                Given two objects and their angular difference, calculates probability the distribution across
                latitude and longitude space.
        Triangulate.match:
                Given one coordinate grid and three 2d probability distributions, calculates the total 
                probability distribution combining the three points.
        
    Outputs:
        Triangulate.lon: 
                Calculated longitude
        Triangulate.lat: 
                Calculated latitude
        Triangulate.errs: 
                The errors of the latitude and longitude [errlon,errlat]
    
    
    """
    def __init__(self):
        self.comp_power = 100
        self.chain=[]
        self.dist=[]
        self.dth=0
        self.dph=0
        self.obserrs = [np.pi/180/60,np.pi/180/60]

    
    def xp(self,a,b):
        """
        Returns x-prime, the vector representing the primed x-axis in the celestial frame. 
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
        """
        Returns y-prime, the vector representing the primed x-axis in the celestial frame. 
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
        """
        Transforms a vector in equatorial celestial frame into an earth alt-az frame at location [a,b].
        
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
            
            
            if th < 0:
                th+=2*np.pi
            
            v_out = np.array([th,ph])
            
        else:
            raise ValueError("Requested representation not understood. Use either 'xyz' or 'th-ph")
            
        return v_out


    def xyz(self,th,ph):
        """Transforms spherical coordinates into xyz coordinates in the celestial frame."""
        return np.array([np.cos(th)*np.cos(ph), np.sin(th)*np.cos(ph), np.sin(ph)])
    
    
    def gen_mock(self,errs=[0,0]):
        """
        Generates a mock triangulation data set, to test the program.
                
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
        """
        Calculates the probability distribution of lattitude and longitude points within the given limits.
        
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
        
        #print obs*np.pi/180
        
            
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
        
        #print thp
        
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
        
        self.chain+=[dist_norm]
        
        return grid,dist_norm
    
    def match(self,grid, c1,c2,c3):
        """
        Combines three probability distributions to isolate points for which the lattitude and longitude 
        values match observation
        
        Inputs
            ------
            c1: ndarray [args,shape=(n,n),dtype=float]
                Normalized distribution as outputted by find_valid.
        
        Outputs:
            stat_array: ndarray [args,shape=(2,2),dtype=float]
                A list consisting in the expected values of the latitude and longitude and the errors.
        """
        
        # We want to compare two functions whose data points are not necessarily the same. Thus, we have to 
        # bin both into a new, global data set.
        
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
        """
        Given v1,v2,v3, equatorial celestial coordinates for three objects, and observed changes in altitude 
        and azimuth of those objects from the ground, returns coordinates of normal vector. i.e. latitude
        and longitude
        
        Input:
            ------
            v1: list [args,shape=(2),dtpe=float]
                The celestial coordinates of the first triangulation object
                
            v2: list [args,shape=(2),dtpe=float]
                The celestial coordinates of the second triangulation object
                
            v3: list [args,shape=(2),dtpe=float]
                The celestial coordinates of the third triangulation object
                
            v2_v1: list [args,shape=(2),dtpe=float]
                The difference in azimuth and altitude between object 2 and object 1
                
            v3_v2: list [args,shape=(2),dtpe=float]
                The difference in azimuth and altitude between object 3 and object 2
        """
        
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
                        
            
        self.lon = av[0]
        self.lat = av[1]
        self.errs = acc
        
        print "Done."
        return av,acc