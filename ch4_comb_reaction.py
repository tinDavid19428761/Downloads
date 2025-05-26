#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2022
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################
"""
Property package for the hydrodealkylation of toluene to form benzene
"""

# Import Python libraries
import logging

# Import Pyomo libraries
from pyomo.environ import (Constraint,
                           exp,
                           Set,
                           Var,
                           Param,
                           units as pyunits)

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        MaterialFlowBasis,
                        ReactionParameterBlock,
                        ReactionBlockDataBase,
                        ReactionBlockBase)
from idaes.core.util.constants import Constants as const
from idaes.core.util.misc import add_object_reference

# Set up logger
_log = logging.getLogger(__name__)


@declare_process_block_class("CH4CombReactionParameterBlock")
class CH4CombReactionParameterData(ReactionParameterBlock):
    """
    Property Parameter Block Class
    Contains parameters and indexing sets associated with properties for
    superheated steam.
    """

    def build(self):
        '''
        Callable method for Block construction.
        '''
        super(CH4CombReactionParameterData, self).build()

        self._reaction_block_class = CH4ReactionBlock

        # List of valid phases in property package
        self.phase_list = Set(initialize=['Vap'])

        # Component list - a list of component identifiers
        self.component_list = Set(initialize=['H2O',
                                              'CO2',
                                              'CH4',
                                              'O2'])

        # Reaction Index
        self.rate_reaction_idx = Set(initialize=["R1"])

        # Reaction Stoichiometry
        self.rate_reaction_stoichiometry = {("R1", "Vap", "H2O"): 2,
                                            ("R1", "Vap", "CO2"): 1,
                                            ("R1", "Vap", "CH4"): -1,
                                            ("R1", "Vap", "O2"): -2,
                                            ("R1", "Liq", "CO2"): 0,
                                            ("R1", "Liq", "CH4"): 0,
                                            ("R1", "Liq", "H2O"): 0,
                                            ("R1", "Liq", "O2"): 0}

        # Arrhenius Constant
        self.arrhenius = Var(initialize=9.90E+08,
                             units=pyunits.mol*pyunits.m**-3*pyunits.s**-1*pyunits.Pa**-1,
                             doc="Arrhenius pre-exponential factor")
        self.arrhenius.fix()

        # Activation Energy
        self.energy_activation = Var(initialize=185476,
                                     units=pyunits.J/pyunits.mol,
                                     doc="Activation energy")
        self.energy_activation.fix()

        # Heat of Reaction
        #engineering toolbox
        dh_rxn_dict = {"R1": -50000}
        self.dh_rxn = Param(self.rate_reaction_idx,
                            initialize=dh_rxn_dict,
                            units=pyunits.J/pyunits.mol,
                            doc="Heat of reaction")

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({
                'k_rxn': {'method': None, 'units': 'm^3/mol.s'},
                'reaction_rate': {'method': None, 'units': 'mol/m^3.s'}
                })
        obj.add_default_units({'time': pyunits.s,
                               'length': pyunits.m,
                               'mass': pyunits.kg,
                               'amount': pyunits.mol,
                               'temperature': pyunits.K})


class ReactionBlock(ReactionBlockBase):
    """
    This Class contains methods which should be applied to Reaction Blocks as a
    whole, rather than individual elements of indexed Reaction Blocks.
    """
    def initialize(blk, outlvl=0, **kwargs):
        '''
        Initialization routine for reaction package.
        Keyword Arguments:
            outlvl : sets output level of initialization routine
                     * 0 = no output (default)
                     * 1 = report after each step
        Returns:
            None
        '''
        if outlvl > 0:
            _log.info('{} Initialization Complete.'.format(blk.name))


@declare_process_block_class("CH4ReactionBlock",
                             block_class=ReactionBlock)
class CH4ReactionBlockData(ReactionBlockDataBase):
    """
    An example reaction package for saponification of ethyl acetate
    """

    def build(self):
        """
        Callable method for Block construction
        """
        super(CH4ReactionBlockData, self).build()

        # Heat of reaction - no _ref as this is the actual property
        add_object_reference(
                self,
                "dh_rxn",
                self.config.parameters.dh_rxn)

        self.k_rxn = Var(initialize=0.2,
                         units=pyunits.mol*pyunits.m**-3*pyunits.s**-1*pyunits.Pa**-1)

        self.reaction_rate = Var(self.params.rate_reaction_idx,
                                 initialize=1,
                                 units=pyunits.mol/pyunits.m**3/pyunits.s)

        self.arrhenus_equation = Constraint(
            expr=self.k_rxn ==
            self.params.arrhenius * exp(
                -self.params.energy_activation /
                (const.gas_constant*self.state_ref.temperature)))

        self.rate_expression = Constraint(
            expr=self.reaction_rate["R1"] ==
            self.k_rxn * self.state_ref.pressure *
            self.state_ref.mole_frac_phase_comp["Vap", "toluene"])

    def get_reaction_rate_basis(b):
        return MaterialFlowBasis.molar