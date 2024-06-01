# Overview of steps:
- Construct the corresponding linear region from an input: input x --> linear constraints (A, b);
- Remove redundant constraints: linear constraints (A, b) --> linear constraints (A, b);
- Linear patch function: linear constraints (A, b), property &Phi;<sub>out</sub> = (P, q_l, q_u), neural network on linear region: f = (f_1, f_2) --> linear patch function(c, d);
- check if P is full row rank; 2. extend P to a full rank square matrix; 3. find the inverse of P; 4. use LP to check the range of z<sub>i</sub>. If it's out of range, perform a linear transform; 5. gather all the linear transforms in a matrix T; 6. cx+d = V<sup>-1</sup>TVf - f.
- Support neural network: linear constraints (A, b), parameter n --> g<sub>A</sub>;
- Local patch neural network: g<sub>A</sub>, linear patch function(c, d) --> h<sub>A</sub>;
