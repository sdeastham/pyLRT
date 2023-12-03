import numpy as np
import subprocess
import io
import os
import tempfile
import xarray as xr


class RadTran():
    '''The base class for handling a LibRadTran instance.
    First intialise and instance, then edit the properties using the 
    'options' directory. 

    Run the radiative transfer code by calling the 'run' method.

    Set the verbose option to retrieve the verbose output from UVSPEC.'''

    def __init__(self, folder, env=None):
        '''Create a radiative transfer object.
        folder - the folder where libradtran was compiled/installed'''
        self.folder = folder
        self.options = {'data_files_path': '../data'}
        self.cloud = None
        self.icecloud = None
        self.atm_profile = None
        self.env = env

    def setup_atmosphere_from_reference(self):
        import pandas as pd
        z_vec = self.atm_profile['z']
        p_vec = self.atm_profile['p']
        T_vec = self.atm_profile['T']
        # Use data from outside the given profile?
        extrap = self.atm_profile['extrap']
        if self.atm_profile['ref'] is not None:
            reference = self.atm_profile['ref']
        else:
            reference = 'afglus.dat'
        headers = ['z(km)','p(mb)','T(K)','air(cm-3)','o3(cm-3)','o2(cm-3)','h2o(cm-3)','co2(cm-3)','no2(cm-3)']
        df = pd.read_csv(os.path.join(self.options['data_files_path'],'atmmod',reference),comment='#',
                         delimiter='\s+',names=headers)

        # After changing temperatures etc, we want the VMRs to be consistent
        vmr = {}
        airden = df['air(cm-3)']
        for spc in ['o3','o2','h2o','co2','no2']:
            vmr[spc] = df[spc + '(cm-3)'] / airden
        # T in K, p in mbar/hPa, z in km
        # Verify that z_vec is monotonically decreasing
        assert np.all(z_vec[1:]<z_vec[:-1]), 'Z data must be monotonically decreasing'
        
        z_ref = df['z(km)'].values

        data_src = {'z(km)': z_vec, 'p(mb)': p_vec, 'T(K)': T_vec}
        if extrap:
            # Take the original z vector and add in layers for the new data
            z_in_min = z_vec[-1]
            if z_in_min <= np.min(z_ref) or np.abs(z_in_min) < 1.0e-5:
                data_below = []
            else:
                i_first = np.argmax(z_ref < z_in_min)
                data_below = list(range(i_first,len(z_ref)))
            z_in_max = z_vec[0]
            if z_in_max > np.max(z_ref):
                data_above = []
            else:
                i_last = np.argmax(z_ref <= z_in_max)
                data_above = list(range(i_last))
            n_below = len(data_below)
            n_above = len(data_above)
            n_new   = len(z_vec)
            full_len = n_above + n_new + n_below
            
            data = {}
            for var, vec_src in data_src.items():
                vec_ref = df[var].values
                vec = np.zeros(n_below+n_new+n_above)
                vec[:n_above] = vec_ref[data_above]
                vec[n_above:(n_above+n_new)] = vec_src
                vec[(n_above + n_new):] = vec_ref[data_below]
                data[var] = vec
        else:
            data = data_src
        
        # Calculate air number density
        boltzmann_constant_k = 1.380649e-23 # J/K
        molec_per_cm3 = 1e-6 * data['p(mb)'] * 1.0e2/(data['T(K)'] * boltzmann_constant_k)
        data['air(cm-3)'] = molec_per_cm3
        
        # Re-derive the species concentrations at the interpolation points
        conc = {}
        z_vec_full = data['z(km)']
        for spc, vmr_vec in vmr.items():
            vmr_interp_vec = np.interp(z_vec_full,z_ref,vmr_vec)
            conc[spc] = vmr_interp_vec * molec_per_cm3
        
        for spc in conc.keys():
            data[spc + '(cm-3)'] = conc[spc]
        df_out = pd.DataFrame(data)
        return df_out

    def run(self, verbose=False, print_input=False, print_output=False, regrid=True, quiet=False, debug=False):
        '''Run the radiative transfer code
        - verbose - retrieves the output from a verbose run, including atmospheric
                    structure and molecular absorption
        - print_input - print the input file used to run libradtran
        - print_output - echo the output
        - regrid - converts verbose output to the regrid/output grid, best to leave as True
        - quiet - if True, do not print UVSPEC warnings
        - debug - dump ALL output (stdout and stderr) to terminal'''
        if self.cloud:  # Create cloud file
            tmpcloud = tempfile.NamedTemporaryFile(delete=False)
            cloudstr = '\n'.join([
                ' {:8.5f} {:7.5f} {:7.5f}'.format( # was 4.2
                #' {:4.2f} {:4.2f} {:4.2f}'.format( # was 4.2
                    self.cloud['z'][alt],
                    self.cloud['lwc'][alt],
                    self.cloud['re'][alt])
                for alt in range(len(self.cloud['lwc']))])
            tmpcloud.write(cloudstr.encode('ascii'))
            tmpcloud.close()
            self.options['wc_file 1D'] = tmpcloud.name

        if self.icecloud:  # Create cloud file
            tmpicecloud = tempfile.NamedTemporaryFile(delete=False)
            icecloudstr = '\n'.join([
                ' {:8.5f} {:7.5f} {:7.5f}'.format( # was 4.2
                #' {:4.2f} {:4.2f} {:4.2f}'.format( # was 4.2
                    self.icecloud['z'][alt],
                    self.icecloud['iwc'][alt],
                    self.icecloud['re'][alt])
                for alt in range(len(self.icecloud['iwc']))])
            tmpicecloud.write(icecloudstr.encode('ascii'))
            tmpicecloud.close()
            self.options['ic_file 1D'] = tmpicecloud.name

        if self.atm_profile:  # Create atmospheric profile of physical properties
            import csv
            # Need z in km, p in hPa/mb, T in K. "ref" is the reference file - set to "None"
            # to just use the US standard atmosphere
            df_atm = self.setup_atmosphere_from_reference()
            # Reformat - assumes that fixed-width is not required
            # Original file gave Z, p, and T as floats, but everything else in exponential notation
            formats = {'z(km)': '{:08.3f}', 'p(mb)': '{:010.5f}', 'T(K)': '{:08.3f}'}
            df_formatted = df_atm.copy()
            for col, f in formats.items():
                df_formatted[col] = df_atm[col].map(lambda x: f.format(x))
            tmpatm = tempfile.NamedTemporaryFile(delete=False)
            df_formatted.to_csv(tmpatm,sep=' ',header=False,index=False,
                                float_format='%10.6E',quoting=csv.QUOTE_NONE)
            tmpatm.close()
            self.options['atmosphere_file'] = tmpatm.name

        if verbose:
            try:
                del(self.options['quiet'])
            except:
                pass
            self.options['verbose'] = ''

        inputstr = '\n'.join(['{} {}'.format(name, self.options[name])
                              for name in self.options.keys()])
        if print_input:
            print(inputstr)
            if self.cloud:
                print('Cloud')
                print('  Alt  LWC   Re')
                print(cloudstr)
            if self.icecloud:
                print('Ice Cloud')
                print('  Alt  IWC   Re')
                print(icecloudstr)
            print('')

        cwd = os.getcwd()
        os.chdir(os.path.join(self.folder, 'bin'))

        process = subprocess.run([os.getcwd()+'/uvspec'], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, env=self.env,
                                 input=inputstr, encoding='ascii')
        os.chdir(cwd)

        if self.cloud:
            os.remove(tmpcloud.name)
            del(self.options['wc_file 1D'])

        if self.icecloud:
            os.remove(tmpicecloud.name)
            del(self.options['ic_file 1D'])

        if debug:
            print('DEBUG: stdout')
            for line in io.StringIO(process.stdout):
                print(line.strip())
            print('DEBUG: stderr')
            for line in io.StringIO(process.stderr):
                print(line.strip())

        # Check uvspec output for errors/warnings
        if not quiet:
            print_flag = False
            for line in io.StringIO(process.stderr):
                if line.startswith('*** Warning'):
                    # Some uvspec warnings start with three stars
                    # These have three stars for every line
                    print(line.strip())
                elif line.startswith('*****'):
                    # Many uvspec warnings are in a star box
                    print(line.strip())
                    print_flag = not(print_flag)
                elif print_flag:
                    # Print line if we are within a star box
                    print(line.strip())

        #Check for errors!
        error = ['UVSpec Error Message:\n']
        error_flag = False
        for line in io.StringIO(process.stderr):
            if line.startswith('Error'):
                error_flag = True
            if error_flag:
                error.append(line)
        if error_flag:
            error = ''.join(error)
            raise ValueError(error)

        if print_output:
            print('Output file:')
            print(process.stdout)
        if verbose:
            try:
                del(self.options['verbose'])
            except:
                pass
            self.options['quiet'] = ''

            return (np.genfromtxt(io.StringIO(process.stdout)),
                    _read_verbose(io.StringIO(process.stderr), regrid=regrid))
        return np.genfromtxt(io.StringIO(process.stdout))


def _skiplines(f, n):
    '''Skip n lines from file f'''
    for i in range(n):
        _ = f.readline()


def _skiplines_title(f, n, t):
    '''Skip n lines from file f. Return the title on line t'''
    for i in range(n):
        if i == t:
            title = [a.strip() for a in f.readline().strip().split('|')]
        _ = f.readline()
    return title


def _match_table(f, start_idstr, nheader_rows, header_row=None):
    '''Get the data from an individual 2D table. 
    start_idstr - string to locate table
    nheader_rows - number of rows to skip before the data'''
    while True:
        line = f.readline()
        if line.startswith(start_idstr):
            break
    title = _skiplines_title(f, nheader_rows, header_row)
    profiles = []
    while True:
        line = f.readline()
        if line.startswith(' --'):
            break
        elif line.startswith('T'):
            break
        else:
            profiles.append(line.replace('|', ''))
    profiles = np.genfromtxt(io.StringIO(''.join(profiles)))
    return title, profiles


def _read_table(f, start_idstr, labels, wavelengths, regrid=False):
    '''Read in the 3D data tables (e.g. optical properties)'''
    optprop = []
    num_wvl = len(wavelengths['wvl'])
    for wv in range(num_wvl):
        temp = _match_table(f, start_idstr, 4, 2)
        optprop.append(temp[1])
        # Could potentially read variable names from the table in future
        #optproplabels = temp[0]
    optprop = np.array(optprop)
    if regrid:
        optprop = _map_to_outputwvl(optprop, wavelengths)
        optprop = xr.Dataset(
            {labels[a]: (['wvl', 'lc'], optprop[:, :, a])
             for a in range(1, optprop.shape[2])},
            coords={'lc': range(optprop.shape[1]),
                    'wvl': np.unique(wavelengths['OutputWVL'].data)})
    else:
        optprop = xr.Dataset(
            {labels[a]: (['wvl', 'lc'], optprop[:, :, a])
             for a in range(1, optprop.shape[2])},
            coords={'lc': range(optprop.shape[1]),
                    'wvl': wavelengths['wvl']})
    return optprop


def _get_wavelengths(f):
    '''Readin the wavelength information'''
    while True:
        line = f.readline()
        if line.startswith(' ... calling setup_rte_wlgrid()'):
            number = int(f.readline().strip().split(' ')[0])
            break
    _skiplines(f, 1)
    wavelengths = []
    for i in range(number):
        wavelengths.append([float(a.strip().replace(' nm', ''))
                            for a in f.readline().strip().split('|')])
    wavelengths = np.array(wavelengths)
    return xr.Dataset(
        {'OutputWVL': (['wvl'], wavelengths[:, 0]),
         'Weights': (['wvl'], wavelengths[:, 2])},
        coords={'wvl': wavelengths[:, 1]})


def _read_verbose(f, regrid=False):
    '''Readin the uotput from 'verbose' to a set of xarrays'''
    try:
        wavelengths = _get_wavelengths(f)
    except:
        print('Readin of verbose file failed.')
        for i in range(10):
            print(f.readline())
        return None

    profiles = _match_table(f, '*** Scaling profiles', 4, 1)
    proflabels = ['lc', 'z', 'p', 'T', 'air', 'o3',
                  'o2', 'h2o', 'co2', 'no2', 'o4']
    profiles = xr.Dataset(
        {proflabels[a]: (['lc'], profiles[1][:, a])
         for a in range(1, profiles[1].shape[1])},
        coords={'lc': range(profiles[1].shape[0])})

    gaslabels = ['lc', 'z', 'rayleigh_dtau', 'mol_abs',
                 'o3', 'o2', 'h2o', 'co2', 'no2', 'bro', 'oclo',
                 'hcho', 'o4', 'so2', 'ch4', 'n2o', 'co', 'n2']

    redistlabels = ['lc', 'z',
                    'o3', 'o2', 'co2', 'no2', 'bro',
                    'oclo', 'hcho', 'wc.dtau', 'ic.dtau']

    optproplabels = ['lc', 'z', 'rayleigh_dtau',
                     'aer_sca', 'aer_abs', 'aer_asy',
                     'wc_sca', 'wc_abs', 'wc_asy',
                     'ic_sca', 'ic_abs', 'ic_asy',
                     'ff', 'g1', 'g2', 'f', 'mol_abs']

    gases = _read_table(f, '*** setup_gases', gaslabels, wavelengths, regrid=regrid)
    redist = _read_table(f, '*** setup_redistribute', redistlabels, wavelengths, regrid=regrid)
    optprop = _read_table(f, '*** optical_properties',
                          optproplabels, wavelengths, regrid=regrid)

    return {'wavelengths': wavelengths,
            'profiles': profiles,
            'gases': gases,
            'redist': redist,
            'optprop': optprop}


def _map_to_outputwvl(data, wavelengths):
    '''Regrids the table properties to the output wavelengths
    
    NOTE: data is still a numpy array'''
    output_wvl = np.array(np.unique(wavelengths['OutputWVL'].data))
    if len(data.shape)>1:
        opdata = np.zeros([output_wvl.shape[0]] + list(data.shape)[1:])
    else:
        opdata = np.zeros(output_wvl.shape[0])
    for i in range(wavelengths['wvl'].shape[0]):
        opdata[np.where(output_wvl==wavelengths['OutputWVL'].data[i])[0]] += wavelengths['Weights'].data[i]*data[i]
    return opdata


