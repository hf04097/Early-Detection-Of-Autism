autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9=< 0, A6 =< 0.50, A2 =< 0.50, A4 > 0.5, A5 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9=< 0.50, A6 =< 0.50, A2 > 0.50, A1 =< 0.5, A8 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9=< 0.50, A6 =< 0.50, A2 > 0.50, A1 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9=< 0.50, A6 > 0.50, A5 =< 0.50,  A2 =< 0.5, A1 > 0.50.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9=< 0.50, A6 > 0.50, A5 =< 0.50,  A2 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9=< 0.50, A6 > 0.50, A5 > 0.50,  A10 =< 0.5, AGE > 21.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9=< 0.50, A6 > 0.50, A5 > 0.50,  A10 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9 > 0.5, A6 =< 0.5, A5 =< 0.50, A2 =< 0.5, A10 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9 > 0.5, A6 =< 0.5, A5 =< 0.50, A2 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9 > 0.5, A6 =< 0.5, A5 > 0.50, AGE =< 12.50, A7 > 0.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9 > 0.5, A6 =< 0.5, A5 > 0.50, AGE > 12.5.

autistic(A1,A2,A3,A4,A5,A6,A7,A8,A9,A10, AGE) :- A9 > 0.5, A6 > 0.5.





