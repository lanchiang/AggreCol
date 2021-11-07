# AggreCol

Algorithm to detect aggregations in verbose CSV files.

## Getting Started

### Installing

* This project is implemented in Python 3.7.9.
* Use the following command to download all required libraries for Python:
```
pip install -r requirements.txt
```

### Executing program

* Use the following script to run the AggreCol code with the default setting:
```
cd scripts/bash/
sh run-aggrecol.sh
```
* Explanation for the parameters in the above command:
  * -d: dataset
  * -o: the function of the aggregations to be detected (All for all functions).
  * -c: coverage threshold
  * -t: timeout
  * -s: until which stage should the algorithm execute.


* Results are stored in the json format in the following file:
```
./results/aggrecol-results.jl
```

[comment]: <> (## Help)

[comment]: <> (Any advise for common problems or issues.)

[comment]: <> (```)

[comment]: <> (command to run if program contains helper info)

[comment]: <> (```)

## Version History

[comment]: <> (* 0.2)

[comment]: <> (    * Various bug fixes and optimizations)

[comment]: <> (    * See [commit change]&#40;&#41; or See [release history]&#40;&#41;)
* 0.1
    * Initial Release

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) License - see the LICENSE.md file for details

## Acknowledgments

* [Gerardo Vitagliano](https://github.com/vitaglianog)
* [Mazhar Hameed](https://github.com/HMazharHameed)
* [Felix Naumann](https://github.com/felix-naumann)
