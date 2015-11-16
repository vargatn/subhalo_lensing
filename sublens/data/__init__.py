"""
I/O and profile maker for measured \Delta\Sigma profiles


Class for easy interface with xshear output and config files, also for
easy selection of appropriate lenses.


Functionality:
---------------

* xshear config maker: creates config and description file for shear calc.
                        (containes name and descriptin of the original data)

* xshear output reader

* lens selector: selects lenses based on their properties listed in the main
                    catalog. This is done by matching the lens ID (0th line)


* shear profile maker: creates shear profile using the entries returned by the
                        lens selector. Also calculates jackknife errors

                        - The profile is saved along with a logfile
                          describing the dataset
                        
"""
