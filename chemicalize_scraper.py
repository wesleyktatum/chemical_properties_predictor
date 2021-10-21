import os
import sys
from glob import glob
import pandas as pd
import zipfile

from rdkit import Chem
from rdkit.Chem import PandasTools

class ChemicalizeExtractor():
    """
    
    """
    def __init__(self, database_path):
        super().__init__()
        
        self.dbase_path = database_path
        self.database = None
        self.struct_handler = MolStructureHandler()

        self.xl_files = ['basic_properties', 'geometry', 'lipophilicity', 'names_and_identifiers',
                         'structural_properties'] #pKa, logD
        self.xl_arrays = ['pKa', 'logD']
        
        self.structure_file = 'structure.sdf'
        
        return

    
    def unzip_downloads(self, zipped_directory = None, extraction_path = None):
        
        #assumes files are in downloads
        if zipped_directory != None:
            zipped_path = zipped_directory
        else:
            zipped_path = 'C:\\Users\\tatum\\Downloads\\calculation-result*.zip'
        
        #default save path is Desktop
        if extraction_path != None:
            extr_path = extraction_path
        else:
            extr_path = 'C:\\Users\\tatum\\Desktop\\unzipped_dirs\\'
        
        #get list of zipped files and save the uncompressed versions
        zipped_dirs = glob(zipped_path)
        for zpd_dir in zipped_dirs:
            with zipfile.ZipFile(zpd_dir, 'r') as f:
                f.extractall(extr_path+zpd_dir[-19:-4])
        
        #get list of unzipped dirs containing chemicalize files
        self.dirs = glob(extr_path+'*')
        
        return
    
    
    def extract_xlsx(self):
        new_additions = []
        
        for unzp in self.dirs:
            molecule = [[],[],[]]
            mol_dict = {}
            for fl in self.xl_files:
#                 print(fl)
                try:
                    addition_df = pd.read_csv(unzp + '\\' + fl + '.csv')

                    prop_names = addition_df['Property'].tolist()
                    values = addition_df['Value'].tolist()
                    units = addition_df['Unit'].tolist()

                    molecule[0].extend(prop_names)
                    molecule[1].extend(values)
                    molecule[2].extend(units)
                    
                except:
                    print(unzp + '\\' + fl + '.csv does not exist')
                
            for i, prop in enumerate(molecule[0]):
                mol_dict[prop] = molecule[1][i]
                mol_dict[prop+' Unit'] = molecule[2][i]
#             print(mol_dict, '\n')
                    
            new_additions.append(mol_dict)
            
#         print(new_additions)
        new_mol_df = pd.DataFrame()
        for mol in new_additions:
#             print(mol)
            mol_df = pd.DataFrame.from_dict(mol, orient = 'index')
            mol_df = mol_df.transpose()
#             print(mol_df)
            new_mol_df = new_mol_df.append(mol_df, ignore_index = True)
            
        return new_mol_df
    
    
    def extract_xl_arrays(self):
        ph_dependent_df = pd.DataFrame()

        for unzp in self.dirs:
            
            mol_df = pd.Series()
            for fl in self.xl_arrays:
                try:
                    if fl == 'pKa':
                        raw_pka = pd.read_csv(unzp + '\\' + fl + '.csv')
                        x, y = raw_pka.shape
                        y = y-1 #w/o pH column, y now equals the # of microspecies
                        pH_vals = raw_pka['pH'].tolist()
                        raw_pka = raw_pka.drop('pH', axis = 1)

                        #want to save all microspecies at each pH as a list, creating a list of lists
                        pka_lists = []
                        for row in raw_pka.iterrows():
                            row_list = []

                            for el in row[1]:
                                row_list.append(el)

                            pka_lists.append(row_list)

                        mol_df['pH_vals'] = pH_vals
                        mol_df['number_of_microspecies'] = y
                        mol_df['pH_dependent_microspecies'] = pka_lists

                    if fl == 'logD':
                        raw_logD = pd.read_csv(unzp + '\\' + fl + '.csv')

                        #want to save all logDs at each pH as a list
                        logD_list = raw_logD['logD'].tolist()

                        mol_df['logD'] = logD_list
                except:
                    print(unzp + '\\' + fl + '.csv does not exist')
                    
            ph_dependent_df = ph_dependent_df.append(mol_df, ignore_index = True)
            
        return ph_dependent_df
                
                    
    
    
    def load_db(self):
        database = pd.read_excel(self.dbase_path)
        self.database = database
        return
    
    
    def save_to_db(self, new_mol_df):
        if self.database is not None:
            pass
        else:
            self.load_db
        
        #check for duplicates and save others
        duplicates = []
        
        #make sure database has InChI column
        if 'InChI' in self.database.columns:
            for inchi in new_mol_df['InChI']:
                if inchi in self.database['InChI'].values:
                    print('Duplicate Compound:\t', inchi)
                    duplicates.append(inchi)
                else:
                    pass
            
        else:
            pass
        
        new_mols = new_mol_df[~new_mol_df['InChI'].isin(duplicates)]
        self.database = self.database.append(new_mols, ignore_index = True)
        
        #for some reason, appending new mols can add columns to front of df. This removes them
        self.database = self.database.loc[:, ~self.database.columns.str.contains('Unnamed')]
        
        self.database.to_excel(self.dbase_path)
                
        return
    
    
    def extract_functional_groups(self, new_xl_df):
        smiles = new_xl_df['SMILES'].tolist()
        
        substructure_df = pd.DataFrame()
        
        for sm in smiles:
            mol = self.struct_handler.mol_from_smiles(sm)
            mol_df = self.struct_handler.extract_functional_groups(mol)
            substructure_df = substructure_df.append(mol_df, ignore_index = True)
            
        return substructure_df
    
    
    def extract(self, return_df = False):
        self.load_db()
        
        #These lines can be used for re-extracting from uncompressed files, otherwise leave them commented out
#         extr_path = 'C:\\Users\\tatum\\Desktop\\unzipped_dirs\\'
#         self.dirs = glob(extr_path+'*')
        
        #extract from the various types of files
        new_xl_df = self.extract_xlsx()
        new_ph_dep_df = self.extract_xl_arrays()
        new_funct_gr_df = self.extract_functional_groups(new_xl_df)
        
        new_mol_df = new_xl_df.join(new_ph_dep_df)
        new_mol_df = new_mol_df.join(new_funct_gr_df)
        
        self.save_to_db(new_mol_df)
        
        if return_df:
            return new_mol_df
        else:
            return
    
    
class MolStructureHandler():
    """
    Class to load, plot, and analyze chemical structures. SMILES or InChI can be used, though SMILES IS 
    PREFERRED as it more cleanly preserves molecular structure. RDKit is used to interpret the structure
    and for sub-structure matching.
    
    Sub-structure matching is currently only supported with SMILES strings
    """
    
    def __init__(self, style = 'SMILES'):
        super().__init__()
        
        self.style = style
        
        #substructure patterns:
        self.functional_groups_smiles = {
            "1,1,1-trifluoroethane": "CC(F)(F)F",
            "1,1'-biphenyl": "C1(C2=CC=CC=C2)=CC=CC=C1",
            "1H-indene": "C1(CC=C2)=C2C=CC=C1",
            "1H-pyrrole": "[NH]1CCCC1",
            "2-butyne": "CC#CC",
            "2-ethyl-1-butanol": "CCC(CC)CO",
            "2-methylpenta-2,3-diene": "CC=C=C(C)C",
            "(E)-1,2-dimethyldiazene": "C/N=N/C",
            "N,N-dimethylacetamide": "CC(N(C)C)=O",
            "N-methylpropan-2-imine": "C/C(C)=N/C",
            "(Z)-N,N,N'-trimethylacetimidamide": "C/C(N(C)C)=N/C",
            "acetic anydride": "CC(=O)OC(=O)C",
            "acyl bromide": "C(=O)Br",
            "acyl chloride": "C(=O)Cl",
            "acyl fluoride": "C(=O)F",
            "acyl iodide": "C(=O)I",
            "aldehyde": "CC=O",
            "amide": "C(=O)N",
            "amino": "*N",
            "azide": "C([N-][N+]#N)",
            "bicyclohexyl": "C1CCCCC1C1CCCCC1",
            "bromine": "Br",
            "but-1-ene": "CCC=C",
            "but-1-yne": "CCC#C",
            "carbon dioxide": "O=C=O",
            "carboxylic acid": "C(=O)O",
            "chlorine": "Cl",
            "chloromethyl methyl ether": "COCCl",
            "deuteroethane": "[2H][CH2]C",
            "dimethyl ether": "COC",
            "diethyl ether": "CCOCC",
            "diisopropyl ether": "CC(C)OC(C)C",
            "diazomethane": "C=[N+]=[N-]",
            "diammonium thiosulfate": "[NH4+].[NH4+].[O-]S(=O)(=O)[S-]",
            "enamine": "N",
            "ethane": "CC",
            "ethanethiol": "CCS",
            "ethanol": "CCO",
            "ethene": "C=C",
            "ether": "COC",
            "ester": "C(=O)OC",
            "fluorine": "F",
            "formaldehyde": "C=O",
            "hydrogen cyanide": "C#N",
            "hydroxide": "[OH-]",
            "hydroxyl amine": "NO",
            "ketone": "CC(=O)C",
            "methane": "C",
            "methanethiol": "CS",
            "methyl acetate": "CC(OC)=O",
            "methyl pyrrole": "CN1CCCC1",
            "methyl tert-butyl ether": "CC(C)(C)OC",
            "nitro": "[N+](=O)[O-]",
            "nitromethane": "C[N+]([O-])=O",
            "pentalene": "C12=CC=CC1=CC=C2",
            "perhydroisoquinoline": "N1CC2CCCC2CC1",
            "phenol": "OC1CCCCC1",
            "phenyl": "C=1(C=CC=CC1)",
            "primary alcohol": "O",
            "primary amine": "N",
            "propan-2-one": "CC(C)=O",
            "propanol": "CCC=O",
            "prop-1-ene": "CC=C",
            "prop-1-yne": "CC#C",
            "pyridine-n-oxide": "O=[N+]1CCCCC1",
            "secondary amine": "NC",
            "spiro[5.5]undecane": "C12(CCCCC1)CCCCC2",
            "sulfoxide": "S(=O)(=O)",
            "tetramethylammonium": "C[N+](C)(C)C",
            "thiol": "S",
            "thiosulfate": "OS(=O)(=S)O",
            "trimethylamine": "CN(C)C",
            "triphenylene": "C1(C=CC=C2)=C2C(C=CC=C3)=C3C4=C1C=CC=C4",
        }

        self.ring_systems_smiles = {
            "anthracene": "C12=CC=CC=C1C=C3C(C=CC=C3)=C2",
            'benzene': 'C1=CC=CC=C1',
            "benzene thiol": "C1=CC=C(C=C1)S",
            "cyclobutadiene": "C1=CC=C1",
            "cyclobutane": "C1CCC1",
            "cycloheptane": "C1CCCCCC1",
            "cyclohexane": "C1CCCCC1",
            "cyclohexa-1,3-diene": "C1=CCCC=C1",
            "cyclohexa-1,4-diene": "C1=CCC=CC1",
            "cyclohexene": "C=1CCCCC=1",
            "cyclopentane": "C1CCCC1",
            "cyclopenta-1,3-diene": "C1=CCC=C1",
            "cyclopropane": "C1CC1",
            "cyclopropene": "C1=CC1",
            'furan': 'C1OC=CC=1',
            'indane': 'C1=CC=CC(CCC2)=C12',
            'indole': 'C12=C(C=CN2)C=CC=C1',
            "naphthalene": "C12=CC=CC=C1C=CC=C2",
            'pyridine': 'C1=CC=NC=C1',
            'pyrrole': 'N1C=CC=C1',
            'thiophene': 'S1C=CC=C1',

        }
        
        return
    
    
    def mol_from_smiles(self, smiles):
#         print(smiles)
        m = Chem.MolFromSmiles(smiles, sanitize = False)
        m.UpdatePropertyCache()
        Chem.SetHybridization(m)
        return m
    
    
    def numbered_strcture_from_smiles(self, smiles):
        mol = self.mol_from_smiles(smiles)
        atoms = mol.GetNumAtoms()
        for idx in range( atoms ):
            mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx()))
                
        return mol
    
    
    def draw_aligned_structure_from_smiles(self, mol_smiles_string, template_smiles_string = None, ax=None):
        if not ax:
            f, ax = plt.subplots()
            
        if not template_smiles_string:
            template_smiles_string = mol_smiles_string

        #generate image of molecule    
        m = Chem.MolFromSmiles(mol_smiles_string, sanitize=False)
        m.UpdatePropertyCache()
        Chem.SetHybridization(m)

        #generate sub-structure that is used for alignment
        t = Chem.MolFromSmiles(template_smiles_string, sanitize=False)
        t.UpdatePropertyCache()
        Chem.SetHybridization(t)
        Chem.AllChem.Compute2DCoords(t)

        #re-draw molecule using substructure alignment
        Chem.AllChem.GenerateDepictionMatching2DStructure(m, t)
        img = Chem.Draw.MolToImage(m)

        return ax.imshow(img, interpolation='bessel')
    
    
    def draw_aligned_structure_from_inchi(self, mol_inchi_string, template_inchi_string = None, ax=None):
        if not ax:
            f, ax = plt.subplots()
            
        if not template_inchi_string:
            template_inchi_string = mol_inchi_string

        #generate image of molecule    
        m = Chem.inchi.MolFromInchi(mol_inchi_string, sanitize=False)
        m.UpdatePropertyCache()
        Chem.SetHybridization(m)

        #generate sub-structure that is used for alignment
        t = Chem.inchi.MolFromInchi(template_inchi_string, sanitize=False)
        t.UpdatePropertyCache()
        Chem.SetHybridization(t)
        Chem.AllChem.Compute2DCoords(t)

        #re-draw molecule using substructure alignment
        Chem.AllChem.GenerateDepictionMatching2DStructure(m, t)
        img = Chem.Draw.MolToImage(m)

        return ax.imshow(img, interpolation='bessel')
        
        
    def get_ring_systems(self, mol, includeSpiro=False):
        """
        identifies all rings in molecule, but does not produce functional group flags. For
        that functionality, use self.get_ring_groups
        """
        ri = mol.GetRingInfo()
        systems = []
        for ring in ri.AtomRings():
            ringAts = set(ring)
            nSystems = []
            for system in systems:
                nInCommon = len(ringAts.intersection(system))
                if nInCommon and (includeSpiro or nInCommon>1):
                    ringAts = ringAts.union(system)
                else:
                    nSystems.append(system)
            nSystems.append(ringAts)
            systems = nSystems
        return systems
    
    
    def count_ring_systems(self, mol):
        systems = self.get_ring_systems(mol)
        return len(systems)
    
    
    def get_functional_groups(self, mol, return_matches = False):
        funct_grp_matches = {}
        
        for group, pattern in self.functional_groups_smiles.items():
            funct_gr = Chem.MolFromSmiles(pattern)
            matches = mol.GetSubstructMatches(funct_gr)
            
            if return_matches:
                funct_grp_matches[group] = matches
            else:
                funct_grp_matches[group] = len(matches)
                    
        return funct_grp_matches
    
    
    def get_ring_groups(self, mol, return_matches = False):
        ring_matches = {}
        
        for group, pattern in self.ring_systems_smiles.items():
            funct_gr = Chem.MolFromSmiles(pattern)
            matches = mol.GetSubstructMatches(funct_gr)
            
            if return_matches:
                ring_matches[group] = matches
            else:
                ring_matches[group] = len(matches)
                    
        return ring_matches
    
    
    def get_stereochemistry(self, mol, return_matches = False):
        stereo_matches = {}
        
        matches = Chem.FindMolChiralCenters(mol, force = True, useLegacyImplementation = False)
        
        if return_matches:
            return matches
        else:
            return len(matches)
        
        
    def extract_functional_groups(self, mol, return_matches = False):
        
        #extract all functional groups and return results as dicts
        funct_grs = self.get_functional_groups(mol, return_matches = return_matches)
        rings = self.get_ring_groups(mol, return_matches = return_matches)
        stereos = self.get_stereochemistry(mol, return_matches = return_matches)
        
        #convert dicts to a single dataframe
        all_grs = {}
        for k, v in funct_grs.items():
            all_grs[k] = v
        for k, v in rings.items():
            all_grs[k] = v
        all_grs['stereo_centers'] = stereos
        
        substruct_df = pd.DataFrame.from_dict(all_grs, orient = 'index')
        substruct_df = substruct_df.transpose()
        
        return substruct_df